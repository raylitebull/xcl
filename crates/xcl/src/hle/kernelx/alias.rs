//! Physical pages that show at every address they are mapped at (`XCL_PAGE_ALIAS=1`).
//!
//! Without it a physical page is kept at one address at a time and moved (copied) when the
//! title maps it elsewhere. With it a block of physical pages is a section (`sections`): the
//! title's reservations of 64 KiB pages are placeholders ("spaces"), and mapping a page there
//! maps a view of its slice, so every address the title mapped it at shows the same memory.
//! The views are guarded (`guards`): a write through one is uploaded at all of them.
//!
//! Each block also has a view of its own (its home, never given to the title) through which
//! XCL reads and writes pages by number (DirectStorage destinations). A retail title never maps
//! a page at its home, so homes are plain address space, not the GPU-visible sparse windows.
//!
//! A page mapped at an address outside a space (a placeholder region the title made in a
//! sparse window) is moved there as before: its bytes are copied there and back again when it
//! is unmapped. While it is away its views show what it held when it left.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows_sys::Win32::Foundation as fnd;
use windows_sys::Win32::System::Memory as mem;

use super::sections::{self, Section, PAGE};
use super::{guards, lazy, page_index, page_number, NEXT_PAGE_INDEX};

/// `XCL_PAGE_ALIAS=1` turns physical page aliasing on (default: off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PAGE_ALIAS").is_ok_and(|v| v.trim() == "1"))
}

struct Block {
    section: Arc<Section>,
    /// XCL's own view of the whole block
    home: usize,
    /// pages not freed yet
    live: usize,
    /// one bit a page: committed (asking Windows walked the rest of a block's uncommitted run –
    /// a retail title mapping 100 000 fresh pages one at a time waited 20 s for its first frame)
    committed: Arc<Vec<AtomicU64>>,
}

#[derive(Default)]
struct State {
    /// first page index → block
    blocks: BTreeMap<usize, Block>,
    /// the title's reservations kept as placeholders: base → size
    spaces: BTreeMap<usize, usize>,
    /// address (64 KiB) in a space → (page number, its section, the page's index in it)
    views: BTreeMap<usize, (usize, Arc<Section>, usize)>,
    /// pages moved to an address outside the spaces: page → address, and address → page
    away: HashMap<usize, usize>,
    away_at: BTreeMap<usize, usize>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Whether any aliased view was ever mapped: `committed_at` sits on the lazy fault path and takes
/// no lock before then.
static ACTIVE: AtomicBool = AtomicBool::new(false);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

impl State {
    /// (block's section, the page's index in it, its address in the home view) of page number `page`
    fn page(&self, page: usize) -> Option<(Arc<Section>, usize, usize)> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range(..=index).next_back()?;
        let i = index - first;
        (i < b.section.pages()).then(|| (b.section.clone(), i, b.home + i * PAGE))
    }

    /// The commit bits of page number `page`'s block and its index there.
    fn bits(&self, page: usize) -> Option<(Arc<Vec<AtomicU64>>, usize)> {
        let index = page_index(page)?;
        let (&first, b) = self.blocks.range(..=index).next_back()?;
        (index - first < b.section.pages()).then(|| (b.committed.clone(), index - first))
    }

    fn in_space(&self, at: usize) -> bool {
        self.spaces.range(..=at).next_back().is_some_and(|(&b, &n)| at < b + n)
    }
}

/// Whether page number `page` is committed (a page never used is not).
fn committed(page: usize) -> bool {
    with(|s| s.bits(page)).is_some_and(|(b, i)| b[i / 64].load(Ordering::Acquire) & (1 << (i % 64)) != 0)
}

/// Page number `page` is committed now.
fn note_committed(page: usize) {
    if let Some((b, i)) = with(|s| s.bits(page)) {
        b[i / 64].fetch_or(1 << (i % 64), Ordering::AcqRel);
    }
}

/// The view at `view` was committed by the lazy handler (its page's first use).
pub fn committed_at(view: usize) {
    if !ACTIVE.load(Ordering::Acquire) {
        return;
    }
    if let Some(page) = with(|s| s.views.get(&(view & !(PAGE - 1))).map(|v| v.0)) {
        note_committed(page);
    }
}

/// A reservation of 64 KiB pages the title will map physical pages into: a space. None if
/// Windows refuses the address space.
pub fn reserve_space(size: usize) -> Option<usize> {
    let len = size.checked_next_multiple_of(PAGE)?;
    // SAFETY: only address space is reserved; `free_space` releases it.
    let at = unsafe { sections::reserve(None, len) }?;
    with(|s| s.spaces.insert(at, len));
    Some(at)
}

/// Whether `p` lies in a space.
pub fn in_space(p: usize) -> bool {
    enabled() && with(|s| s.in_space(p))
}

/// `VirtualFree` of an address in a space: release (the whole space) unmaps its pages and gives
/// the address space back; decommit unmaps the pages of the range. None: not a space.
pub unsafe fn free_space(addr: usize, size: usize, t: u32) -> Option<bool> {
    let (base, len) = with(|s| s.spaces.range(..=addr).next_back().filter(|(&b, &n)| addr < b + n).map(|(&b, &n)| (b, n)))?;
    if t & mem::MEM_RELEASE != 0 {
        if addr != base || size != 0 {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return Some(false);
        }
        let part = super::source_part(base, len);
        unmap_range(base, len);
        with(|s| s.spaces.remove(&base));
        super::source_part_going(part, true, || {
            super::mirrors::released(base);
            sections::release_all(base, len);
        });
        return Some(true);
    }
    let end = if size == 0 { base + len } else { addr.saturating_add(size).min(base + len) };
    unmap_range(addr & !(PAGE - 1), end - (addr & !(PAGE - 1)));
    Some(true)
}

unsafe fn unmap_range(at: usize, len: usize) {
    let views: Vec<usize> = with(|s| {
        let keys: Vec<usize> = s.views.range(at..at + len).map(|(&a, _)| a).collect();
        for &a in &keys {
            s.views.remove(&a);
        }
        keys
    });
    if views.is_empty() {
        return;
    }
    for &a in &views {
        guards::forget(a);
        lazy::unmark(a);
    }
    let lo = *views.first().unwrap();
    let hi = *views.last().unwrap() + PAGE;
    super::source_going(lo, hi - lo, false, || {
        for &a in &views {
            sections::unmap(a);
        }
    });
}

/// `count` physical pages for the title; their numbers in `arr`.
pub unsafe fn allocate(n: usize, arr: *mut usize) -> bool {
    let Some(section) = Section::new(n) else {
        crate::xlog_warn!("physical pages: a section of {n} pages could not be made");
        return false;
    };
    let Some(home) = sections::reserve(None, n * PAGE) else { return false };
    if !section.map(0, n, home) {
        sections::release(home);
        return false;
    }
    let first = NEXT_PAGE_INDEX.fetch_add(n, Ordering::Relaxed);
    for i in 0..n {
        *arr.add(i) = page_number(first + i);
    }
    let committed = Arc::new((0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect());
    with(|s| s.blocks.insert(first, Block { section: Arc::new(section), home, live: n, committed }));
    true
}

/// The title frees `pages`. Addresses that show one keep showing it (the title is meant to unmap
/// first); a block whose pages are all freed loses its home.
pub unsafe fn free(pages: &[usize]) {
    let mut homes = Vec::new();
    with(|s| {
        for &page in pages {
            if let Some(at) = s.away.remove(&page) {
                s.away_at.remove(&at);
            }
            let Some(index) = page_index(page) else { continue };
            let Some((&first, b)) = s.blocks.range_mut(..=index).next_back() else { continue };
            if index >= first + b.section.pages() {
                continue;
            }
            b.live = b.live.saturating_sub(1);
            if b.live == 0 {
                homes.push((b.home, b.section.pages() * PAGE));
                s.blocks.remove(&first);
            }
        }
    });
    for (home, len) in homes {
        super::source_going(home, len, true, || {
            sections::unmap(home);
            sections::release(home);
        });
    }
}

/// Map `n` pages (`pages[i]`, 0: unmap) at `va`. Pages at an address in a space are views of
/// them; elsewhere they are moved there. False with the last error set if a page is not one of
/// the title's or Windows refuses.
pub unsafe fn map(va: usize, pages: &[usize]) -> bool {
    if !va.is_multiple_of(PAGE) {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return false;
    }
    if pages.is_empty() {
        return true;
    }
    let end = match pages.len().checked_mul(PAGE).and_then(|n| va.checked_add(n)) {
        Some(e) => e,
        None => {
            fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
            return false;
        }
    };
    // Validate every non-zero page and pull out any existing views in `[va, end)` under one lock
    // so unmapping a batch of views takes a single `source_going` / `sparse::STATE` round-trip.
    let (valid, old_views) = with(|s| {
        if pages.iter().any(|&p| p != 0 && s.page(p).is_none()) {
            return (false, Vec::new());
        }
        let keys: Vec<usize> = s.views.range(va..end).map(|(&a, _)| a).collect();
        for &a in &keys {
            s.views.remove(&a);
        }
        (true, keys)
    });
    if !valid {
        fnd::SetLastError(fnd::ERROR_INVALID_PARAMETER);
        return false;
    }
    if !old_views.is_empty() {
        for &a in &old_views {
            guards::forget(a);
            lazy::unmark(a);
        }
        let lo = *old_views.first().unwrap();
        let hi = *old_views.last().unwrap() + PAGE;
        super::source_going(lo, hi - lo, false, || {
            for &a in &old_views {
                sections::unmap(a);
            }
        });
    }
    for (i, &page) in pages.iter().enumerate() {
        let at = va + i * PAGE;
        if !map_one(at, page) {
            fnd::SetLastError(fnd::ERROR_INVALID_ADDRESS);
            return false;
        }
    }
    true
}

unsafe fn map_one(at: usize, page: usize) -> bool {
    // what was at `at` goes first (any prior view in `[va, end)` was already unmapped by `map`)
    let (had_away, old_away, info, in_sp) = with(|s| {
        let had_away = s.away_at.contains_key(&at);
        if page == 0 {
            return (had_away, None, None, false);
        }
        let old_away = s.away.get(&page).copied();
        let info = s.page(page);
        let in_sp = s.in_space(at);
        (had_away, old_away, info, in_sp)
    });
    if had_away {
        unmap_away(at);
    }
    if page == 0 {
        return true;
    }
    // a page moved elsewhere comes home first (it is at one address outside the spaces at most,
    // as before): its views then show what it holds
    if let Some(old) = old_away {
        unmap_away(old);
    }
    let (section, index, home) = info.unwrap();
    if in_sp {
        if !section.map(index, 1, at) {
            return false;
        }
        let is_committed = committed(page);
        ACTIVE.store(true, Ordering::Release);
        with(|s| s.views.insert(at, (page, section, index)));
        guards::watch_with(page, at, is_committed);
        if is_committed {
            // an already-committed page mapped at a new view: sparse / title_textures must see it
            guards::written(page);
        } else {
            // a page never used: committed at its first touch here (or through its home)
            lazy::mark(at);
        }
        return true;
    }
    move_to(at, page, home)
}

/// Page `page` moved to `at`, an address outside the spaces: committed there with its bytes.
unsafe fn move_to(at: usize, page: usize, home: usize) -> bool {
    let u = crate::gpu::uma::get();
    let ok = if u.is_some_and(|u| u.is_committed(at)) {
        true
    } else if u.is_some_and(|u| u.is_sparse(at)) {
        !super::commit_sparse(at, PAGE, mem::PAGE_READWRITE).is_null()
    } else {
        !mem::VirtualAlloc(at as *const c_void, PAGE, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null()
    };
    if !ok {
        return false;
    }
    if committed(page) {
        std::ptr::copy_nonoverlapping(home as *const u8, at as *mut u8, PAGE);
    } else {
        std::ptr::write_bytes(at as *mut u8, 0, PAGE);
    }
    with(|s| {
        s.away.insert(page, at);
        s.away_at.insert(at, page);
    });
    true
}

/// The page moved to `at` (if any) goes home with what `at` holds; `at` is decommitted.
unsafe fn unmap_away(at: usize) {
    let Some((page, home)) = with(|s| {
        let page = s.away_at.remove(&at)?;
        s.away.remove(&page);
        Some((page, s.page(page)?.2))
    }) else {
        return;
    };
    if sections::commit(home, PAGE) {
        note_committed(page);
        std::ptr::copy_nonoverlapping(at as *const u8, home as *mut u8, PAGE);
        guards::written(page);
    }
    let u = crate::gpu::uma::get();
    if u.is_some_and(|u| u.is_committed(at)) {
        return;
    }
    if u.is_some_and(|u| u.is_sparse(at)) {
        crate::hle::d3d12x::sparse::decommitted(at, PAGE);
    }
    super::source_going(at, PAGE, false, || mem::VirtualFree(at as *mut c_void, PAGE, mem::MEM_DECOMMIT));
}

/// Whether a physical page is mapped at the page holding `va` (a view, or a page moved there).
pub fn physical_page_at(va: usize) -> bool {
    let at = va & !(PAGE - 1);
    with(|s| s.views.contains_key(&at) || s.away_at.contains_key(&at))
}

/// Whether `page` is a physical page allocated now.
pub fn page_live(page: usize) -> bool {
    with(|s| s.page(page).is_some())
}

/// The physical page mapped at the page holding `va` (a view, or a page moved there), or None.
pub fn page_at(va: usize) -> Option<usize> {
    let at = va & !(PAGE - 1);
    with(|s| s.views.get(&at).map(|v| v.0).or_else(|| s.away_at.get(&at).copied()))
}

/// Current storage for a page in the GPU-view mapping record. Section-backed aliases
/// all share the block's home. Ordinary sparse reservations can temporarily own its bytes.
pub(super) fn backing(page: usize) -> Option<(usize, bool)> {
    with(|s| s.away.get(&page).map(|&at| (at, false)).or_else(|| s.page(page).map(|(_, _, home)| (home, true))))
}

/// The spaces and the blocks' homes as (base, bytes), each by address (`mem_map`).
pub fn ranges() -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let (spaces, mut homes): (Vec<_>, Vec<_>) = with(|s| (s.spaces.iter().map(|(&b, &n)| (b, n)).collect(), s.blocks.values().map(|b| (b.home, b.section.pages() * PAGE)).collect()));
    homes.sort_unstable();
    (spaces, homes)
}

/// What one call of the queue's page-mapping method did, in 64 KiB pages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// destination pages that show a physical page now
    pub mapped: u64,
    /// destination pages unmapped (by the call, or because nothing was mapped at their pool page)
    pub unmapped: u64,
    /// of those, pages whose pool page had nothing mapped
    pub empty_pool: u64,
}

/// The queue's page-mapping method (`CopyPageMappingsBatchX`, decoded by
/// `mirrors::decode_batches`): each destination page shows the physical page mapped at its pool
/// page now – on the console the method copies the pool page's mapping – or nothing.
pub unsafe fn apply_page_mappings(changes: &[(usize, Option<usize>)]) -> Option<Applied> {
    let mut out = Applied::default();
    for &(dest, pool) in changes {
        let page = pool.and_then(page_at);
        if !map_one(dest & !(PAGE - 1), page.unwrap_or(0)) {
            return None;
        }
        match (pool, page) {
            (_, Some(_)) => out.mapped += 1,
            (Some(_), None) => {
                out.unmapped += 1;
                out.empty_pool += 1;
            }
            (None, _) => out.unmapped += 1,
        }
    }
    Some(out)
}

/// Where page `page`'s bytes are now (committing its home if `write`), or None.
unsafe fn bytes_of(page: usize, write: bool) -> Option<(usize, bool)> {
    let (away, home) = with(|s| (s.away.get(&page).copied(), s.page(page).map(|p| p.2)));
    if let Some(at) = away {
        return Some((at, true));
    }
    let home = home?;
    if committed(page) {
        return Some((home, true));
    }
    if write {
        if !sections::commit(home, PAGE) {
            return None;
        }
        note_committed(page);
        return Some((home, true));
    }
    Some((home, false))
}

/// `read_physical_pages` in alias mode.
pub unsafe fn read(pages: &[usize], offset: usize, dst: &mut [u8]) -> bool {
    copy_pages(pages, offset, dst.len(), |addr, done, n| {
        match addr {
            Some(a) => std::ptr::copy_nonoverlapping(a as *const u8, dst.as_mut_ptr().add(done), n),
            None => std::ptr::write_bytes(dst.as_mut_ptr().add(done), 0, n),
        }
    }, false)
}

/// `write_physical_pages` in alias mode.
pub unsafe fn write(pages: &[usize], offset: usize, src: &[u8]) -> bool {
    copy_pages(pages, offset, src.len(), |addr, done, n| {
        if let Some(a) = addr {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(done), a as *mut u8, n);
        }
    }, true)
}

/// Walk `len` bytes from `offset` in `pages`, calling `f(address or None for a page never used,
/// bytes done, bytes here)`. All pages are checked before any byte is touched.
unsafe fn copy_pages(pages: &[usize], offset: usize, len: usize, mut f: impl FnMut(Option<usize>, usize, usize), write: bool) -> bool {
    if len == 0 {
        return offset < PAGE;
    }
    let Some(need) = offset.checked_add(len).map(|end| end.div_ceil(PAGE)) else { return false };
    if offset >= PAGE || pages.len() < need {
        return false;
    }
    let mut addrs = Vec::with_capacity(need);
    for &p in &pages[..need] {
        let Some(a) = bytes_of(p, write) else { return false };
        addrs.push((p, a));
    }
    let (mut done, mut off) = (0, offset);
    for (page, (addr, used)) in addrs {
        let n = (PAGE - off).min(len - done);
        f(used.then_some(addr + off), done, n);
        if write && with(|s| !s.away.contains_key(&page)) {
            guards::written(page);
        }
        done += n;
        off = 0;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// alias state and guards are one per process: these tests run one at a time, and not
    /// beside the guards' own
    use super::guards::tests::SERIAL;

    unsafe fn read64(at: usize) -> u64 {
        std::ptr::read_volatile(at as *const u64)
    }

    unsafe fn write64(at: usize, v: u64) {
        std::ptr::write_volatile(at as *mut u64, v)
    }

    fn pages(n: usize) -> Vec<usize> {
        let mut arr = vec![0usize; n];
        assert!(unsafe { allocate(n, arr.as_mut_ptr()) });
        arr
    }

    fn dirty() -> Vec<usize> {
        let mut out = Vec::new();
        guards::take_dirty(&mut out, PAGE);
        out.sort_unstable();
        out
    }

    #[test]
    fn a_page_mapped_into_two_spaces_is_one_page_and_writes_reach_both() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(4);
        let cpu = reserve_space(8 * PAGE).unwrap();
        let gpu = reserve_space(8 * PAGE).unwrap();
        unsafe {
            assert!(map(cpu + 2 * PAGE, &p[1..3]));
            assert!(map(gpu + 5 * PAGE, &[p[2]]));
            assert!(physical_page_at(gpu + 5 * PAGE + 123));
            write64(cpu + 3 * PAGE + 8, 0xABCD); // page p[2] through the first space
            assert_eq!(read64(gpu + 5 * PAGE + 8), 0xABCD, "the other space shows the same page");
            dirty();
            write64(gpu + 5 * PAGE, 1);
            let mut want = vec![cpu + 3 * PAGE, gpu + 5 * PAGE];
            want.sort_unstable();
            assert_eq!(dirty(), want, "a write through one view: both are uploaded");
            assert_eq!(free_space(cpu, 0, mem::MEM_RELEASE), Some(true));
            assert_eq!(read64(gpu + 5 * PAGE + 8), 0xABCD, "releasing one space leaves the page at the other");
            assert_eq!(free_space(gpu, 0, mem::MEM_RELEASE), Some(true));
            free(&p);
        }
    }

    #[test]
    fn remapping_an_address_shows_the_new_page_and_keeps_the_old_one() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let s = reserve_space(2 * PAGE).unwrap();
        unsafe {
            assert!(map(s, &[p[0]]));
            write64(s, 11);
            assert!(map(s, &[p[1]]));
            assert_eq!(read64(s), 0, "a page never written reads zeros");
            assert!(map(s, &[0]));
            assert!(!physical_page_at(s));
            assert!(map(s + PAGE, &[p[0]]));
            assert_eq!(read64(s + PAGE), 11, "the first page kept what was written to it");
            free_space(s, 0, mem::MEM_RELEASE);
            free(&p);
        }
    }

    #[test]
    fn pages_read_and_written_by_number_are_the_pages_the_title_sees() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(3);
        let s = reserve_space(3 * PAGE).unwrap();
        unsafe {
            assert!(map(s, &p));
            dirty();
            // a DirectStorage destination: 16 bytes across the boundary of the first two pages
            assert!(write(&p, PAGE - 8, &[7u8; 16]));
            assert_eq!(read64(s + PAGE - 8), 0x0707_0707_0707_0707);
            assert_eq!(read64(s + PAGE), 0x0707_0707_0707_0707);
            assert_eq!(dirty(), vec![s, s + PAGE], "pages written by number are uploaded where they are mapped");
            write64(s + 2 * PAGE + 16, 5);
            let mut back = [0u8; 8];
            assert!(read(&p[2..], 16, &mut back));
            assert_eq!(u64::from_le_bytes(back), 5);
            let mut fresh = [1u8; 4];
            let q = pages(1);
            assert!(read(&q, 0, &mut fresh));
            assert_eq!(fresh, [0; 4], "a page never used reads zeros without being committed");
            assert!(!read(&p, PAGE, &mut back), "an offset past the first page is refused");
            free_space(s, 0, mem::MEM_RELEASE);
            free(&p);
            free(&q);
        }
    }

    #[test]
    fn a_page_moved_outside_the_spaces_goes_and_comes_back_with_its_bytes() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(1);
        let s = reserve_space(PAGE).unwrap();
        // address space outside the spaces (as a title's own placeholder region would be)
        let out = unsafe { mem::VirtualAlloc(std::ptr::null(), PAGE, mem::MEM_RESERVE, mem::PAGE_READWRITE) } as usize;
        unsafe {
            assert!(map(s, &p));
            write64(s, 21);
            assert!(map(out, &p));
            assert_eq!(read64(out), 21, "moved with its bytes");
            assert!(physical_page_at(out));
            write64(out, 22);
            let mut b = [0u8; 8];
            assert!(read(&p, 0, &mut b));
            assert_eq!(u64::from_le_bytes(b), 22, "read by number where it is now");
            assert!(map(out, &[0]));
            assert_eq!(read64(s), 22, "back home with what it held there");
            free_space(s, 0, mem::MEM_RELEASE);
            mem::VirtualFree(out as *mut c_void, 0, mem::MEM_RELEASE);
            free(&p);
        }
    }

    #[test]
    fn the_queue_page_mapping_method_shows_the_pool_pages_physical_page_in_the_gpu_view() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let p = pages(2);
        let pool = reserve_space(4 * PAGE).unwrap();
        let cpu = reserve_space(4 * PAGE).unwrap();
        let gpu = reserve_space(4 * PAGE).unwrap();
        unsafe {
            // a streamed page: in the pool and in the CPU view, written through the CPU view
            assert!(map(pool + PAGE, &[p[0]]));
            assert!(map(cpu + 2 * PAGE, &[p[0]]));
            write64(cpu + 2 * PAGE, 0x5EED);
            // kind 0 / 1: pool pages at GPU-view pages; pool page 3 holds nothing; kind 2: unmapped
            let changes = [(gpu, Some(pool + PAGE)), (gpu + PAGE, Some(pool + PAGE)), (gpu + 2 * PAGE, Some(pool + 3 * PAGE)), (gpu + 3 * PAGE, None)];
            let done = apply_page_mappings(&changes).unwrap();
            assert_eq!(done, Applied { mapped: 2, unmapped: 2, empty_pool: 1 });
            assert_eq!(read64(gpu), 0x5EED, "the GPU view shows the page the CPU view wrote");
            assert_eq!(read64(gpu + PAGE), 0x5EED, "one pool page at two GPU-view pages");
            assert!(!physical_page_at(gpu + 2 * PAGE) && !physical_page_at(gpu + 3 * PAGE));
            dirty();
            write64(gpu + PAGE + 8, 1);
            let mut want = vec![pool + PAGE, cpu + 2 * PAGE, gpu, gpu + PAGE];
            want.sort_unstable();
            assert_eq!(dirty(), want, "a write in the GPU view: every view of the page is uploaded");
            // the pool page is given another physical page later: the GPU view keeps the one it was given
            assert!(map(pool + PAGE, &[p[1]]));
            assert_eq!(read64(gpu), 0x5EED);
            assert_eq!(page_at(gpu), Some(p[0]));
            for s in [pool, cpu, gpu] {
                free_space(s, 0, mem::MEM_RELEASE);
            }
            free(&p);
        }
    }

    #[test]
    fn a_page_not_from_the_title_is_refused() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let s = reserve_space(PAGE).unwrap();
        unsafe {
            assert!(!map(s, &[page_number(1 << 40)]));
            assert!(!map(s + 4, &[0]), "an address not on a page boundary is refused");
            free_space(s, 0, mem::MEM_RELEASE);
        }
    }
}
