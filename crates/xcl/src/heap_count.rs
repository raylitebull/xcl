//! XCL's own heap, counted (`XCL_HEAP_COUNT=1`, with `XCL_ACTIVITY`): how much of the process's
//! commit is memory XCL's own code allocated, by the size of the allocations, and from where:
//! every allocation of 64 KiB or more with its call stack, and one smaller allocation in
//! `SAMPLE` (chosen by its address) standing for `SAMPLE` times its bytes.
//!
//! The process holds the title's memory, the driver's, the shader compiler's and XCL's own in one
//! number (`memory: N MB committed by the process`). The title's is found by its addresses;
//! this tells XCL's apart from the rest. Every allocation of XCL's Rust code goes through
//! `Counting` (the system allocator with counters around it). Off, it costs one relaxed load an
//! allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};

/// The allocator of the crate: the system's, counted while `on()`.
pub struct Counting;

/// 0: not asked yet, 1: off, 2: on
static STATE: AtomicU8 = AtomicU8::new(0);

/// upper bounds of the size classes, the last one open
const CLASSES: [usize; 7] = [256, 4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20, usize::MAX];
const CLASS_NAMES: [&str; 7] = ["<256 B", "<4 KiB", "<64 KiB", "<1 MiB", "<16 MiB", "<256 MiB", "larger"];
/// live bytes and live allocations of each class
static BYTES: [AtomicUsize; 7] = [const { AtomicUsize::new(0) }; 7];
static COUNT: [AtomicUsize; 7] = [const { AtomicUsize::new(0) }; 7];

/// allocations of this many bytes or more are kept with where they were made (`SITES`)
const LARGE: usize = 64 << 10;
/// one smaller allocation in this many is kept with where it was made (by its address: the
/// free of an allocation not kept costs nothing)
const SAMPLE: usize = 256;
/// frames kept of a kept allocation's call stack (the allocator's own left out)
const FRAMES: usize = 12;
const SLOTS: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Site {
    /// the allocation (0: free slot, 1: a slot freed, passed over in a search)
    ptr: usize,
    size: usize,
    frames: [usize; FRAMES],
}

/// The large allocations alive, an open-addressed table by address (no allocation of its own:
/// it is used inside the allocator).
struct Sites(std::cell::UnsafeCell<[Site; SLOTS]>);
// SAFETY: only touched under `LOCK`
unsafe impl Sync for Sites {}
static SITES: Sites = Sites(std::cell::UnsafeCell::new([Site { ptr: 0, size: 0, frames: [0; FRAMES] }; SLOTS]));
static LOCK: AtomicBool = AtomicBool::new(false);
/// large allocations the full table had no slot for
static UNTRACKED: AtomicUsize = AtomicUsize::new(0);

fn class(size: usize) -> usize {
    CLASSES.iter().position(|&c| size < c).unwrap_or(CLASSES.len() - 1)
}

/// Whether an allocation of `size` bytes at `ptr` is kept with its call stack.
fn kept(ptr: usize, size: usize) -> bool {
    size >= LARGE || ((ptr >> 4).wrapping_mul(0xD6E8_FEB8_6659_FD93) >> (usize::BITS - SAMPLE.trailing_zeros())) == 0
}

/// The bytes a kept allocation stands for.
fn weight(size: usize) -> usize {
    if size >= LARGE { size } else { size * SAMPLE }
}

fn slot_of(ptr: usize) -> usize {
    (ptr >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - SLOTS.trailing_zeros())
}

fn locked<R>(f: impl FnOnce(&mut [Site; SLOTS]) -> R) -> R {
    while LOCK.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
    }
    // SAFETY: `LOCK` is held
    let r = f(unsafe { &mut *SITES.0.get() });
    LOCK.store(false, Ordering::Release);
    r
}

/// Whether allocations are counted (`XCL_HEAP_COUNT=1`). The variable is read without the
/// standard library's `env` (it allocates: this runs inside the allocator).
fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let mut buf = [0u16; 4];
            const NAME: [u16; 15] = {
                let s = b"XCL_HEAP_COUNT\0";
                let mut w = [0u16; 15];
                let mut i = 0;
                while i < 15 {
                    w[i] = s[i] as u16;
                    i += 1;
                }
                w
            };
            let n = unsafe { windows_sys::Win32::System::Environment::GetEnvironmentVariableW(NAME.as_ptr(), buf.as_mut_ptr(), buf.len() as u32) };
            let yes = n == 1 && buf[0] == b'1' as u16;
            STATE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
    }
}

fn added(ptr: *mut u8, size: usize) {
    if ptr.is_null() {
        return;
    }
    let c = class(size);
    BYTES[c].fetch_add(size, Ordering::Relaxed);
    COUNT[c].fetch_add(1, Ordering::Relaxed);
    if kept(ptr as usize, size) {
        let mut frames = [0usize; FRAMES];
        // (skips this function and the allocator's method)
        unsafe {
            windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace(2, FRAMES as u32, frames.as_mut_ptr() as *mut *mut std::ffi::c_void, std::ptr::null_mut());
        }
        let kept = locked(|t| {
            let first = slot_of(ptr as usize);
            for k in 0..64 {
                let s = &mut t[(first + k) & (SLOTS - 1)];
                if s.ptr <= 1 {
                    *s = Site { ptr: ptr as usize, size, frames };
                    return true;
                }
            }
            false
        });
        if !kept {
            UNTRACKED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn removed(ptr: *mut u8, size: usize) {
    let c = class(size);
    BYTES[c].fetch_sub(size, Ordering::Relaxed);
    COUNT[c].fetch_sub(1, Ordering::Relaxed);
    if kept(ptr as usize, size) {
        locked(|t| {
            let first = slot_of(ptr as usize);
            for k in 0..64 {
                let s = &mut t[(first + k) & (SLOTS - 1)];
                if s.ptr == ptr as usize {
                    s.ptr = 1;
                    return;
                }
                if s.ptr == 0 {
                    return;
                }
            }
        });
    }
}

// SAFETY: every call goes to the system allocator unchanged; the counters only watch
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if on() {
            added(p, l.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(l);
        if on() {
            added(p, l.size());
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if on() {
            removed(p, l.size());
        }
        System.dealloc(p, l)
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        let q = System.realloc(p, l, new_size);
        if on() && !q.is_null() {
            removed(p, l.size());
            added(q, new_size);
        }
        q
    }
}

/// The title's allocations XCL serves from the process heap (`kernelx::XMemAllocDefault`):
/// live bytes and live allocations. They share that heap with XCL's own.
static TITLE: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];

/// Whether the counters run (`XCL_HEAP_COUNT=1`): callers that count for others ask first.
pub fn counting() -> bool {
    on()
}

/// The title got `bytes` from the process heap.
pub fn title_added(bytes: usize) {
    TITLE[0].fetch_add(bytes, Ordering::Relaxed);
    TITLE[1].fetch_add(1, Ordering::Relaxed);
}

/// The title gave `bytes` of the process heap back.
pub fn title_removed(bytes: usize) {
    TITLE[0].fetch_sub(bytes, Ordering::Relaxed);
    TITLE[1].fetch_sub(1, Ordering::Relaxed);
}

/// What the process heap holds, by a walk of it under its lock: (bytes allocated, bytes
/// committed, milliseconds the walk took). XCL's own allocations, the title's from
/// `XMemAllocDefault` and those of the driver, Direct3D and the C runtime are all in it.
fn process_heap() -> Option<(usize, usize, u128)> {
    use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSummary, HEAP_SUMMARY};
    let t = std::time::Instant::now();
    let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
    s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
    (unsafe { HeapSummary(GetProcessHeap(), 0, &mut s) } != 0).then(|| (s.cbAllocated, s.cbCommitted, t.elapsed().as_millis()))
}

/// Windows heaps XCL made for someone else's allocations (the shader compiler's, one a worker):
/// (whose, handle). They are not in the counters above, and a heap keeps memory it once used.
static OTHER_HEAPS: std::sync::Mutex<Vec<(&'static str, usize)>> = std::sync::Mutex::new(Vec::new());

/// A serialized Windows heap `whose` allocations go to (`lines` reports what it holds).
pub fn note_heap(whose: &'static str, heap: usize) {
    OTHER_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).push((whose, heap));
}

/// (heaps, bytes allocated in them, bytes they have committed) of the noted heaps of `whose`.
fn other_heaps(whose: &str) -> (usize, usize, usize) {
    use windows_sys::Win32::System::Memory::{HeapSummary, HEAP_SUMMARY};
    let heaps: Vec<usize> = OTHER_HEAPS.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|h| h.0 == whose).map(|h| h.1).collect();
    let (mut allocated, mut committed) = (0, 0);
    for &h in &heaps {
        let mut s: HEAP_SUMMARY = unsafe { std::mem::zeroed() };
        s.cb = std::mem::size_of::<HEAP_SUMMARY>() as u32;
        if unsafe { HeapSummary(h as _, 0, &mut s) } != 0 {
            allocated += s.cbAllocated;
            committed += s.cbCommitted;
        }
    }
    (heaps.len(), allocated, committed)
}

/// (live bytes, live allocations) of XCL's own heap; zeros while not counted.
pub fn live() -> (usize, usize) {
    (BYTES.iter().map(|b| b.load(Ordering::Relaxed)).sum(), COUNT.iter().map(|c| c.load(Ordering::Relaxed)).sum())
}

/// The kept allocations alive, copied into `alive` (made with room for the whole table: no
/// allocation under the lock), but for `alive`'s own buffer, a large allocation of the report's
/// and not XCL's memory being reported.
fn snapshot_into(alive: &mut Vec<Site>) {
    let own = alive.as_ptr() as usize;
    locked(|t| {
        for s in t.iter().filter(|s| s.ptr > 1 && s.ptr != own) {
            alive.push(*s);
        }
    });
}

/// The lines of the activity log: the heap by size class, then the places that hold the most
/// in large allocations, as offsets in xcl.exe (symbolized afterwards from its .pdb). None: off.
pub fn lines(top: usize) -> Option<Vec<String>> {
    if !on() {
        return None;
    }
    let (bytes, count) = live();
    let classes: Vec<String> = (0..CLASSES.len())
        .filter(|&c| COUNT[c].load(Ordering::Relaxed) != 0)
        .map(|c| format!("{} {} MB in {}", CLASS_NAMES[c], BYTES[c].load(Ordering::Relaxed) >> 20, COUNT[c].load(Ordering::Relaxed)))
        .collect();
    let mut out = vec![format!("XCL's own heap: {} MB in {} allocations; by size: {}", bytes >> 20, count, classes.join(", "))];
    let (heaps, allocated, committed) = other_heaps("DXC");
    if heaps != 0 {
        out.push(format!("the shader compiler's heaps: {heaps} heaps, {} MB in use, {} MB committed", allocated >> 20, committed >> 20));
    }
    // (the walk holds the heap's lock: every third time, a minute apart)
    static WALKS: AtomicUsize = AtomicUsize::new(0);
    let title = (TITLE[0].load(Ordering::Relaxed), TITLE[1].load(Ordering::Relaxed));
    match (WALKS.fetch_add(1, Ordering::Relaxed) % 3 == 0).then(process_heap).flatten() {
        Some((allocated, committed, ms)) => out.push(format!(
            "the process heap: {} MB in use, {} MB committed (walked in {ms} ms); of it XCL's own {} MB, the title's {} MB in {} (XMemAllocDefault), others' (driver, Direct3D, C runtime) {} MB",
            allocated >> 20,
            committed >> 20,
            bytes >> 20,
            title.0 >> 20,
            title.1,
            allocated.saturating_sub(bytes + title.0) >> 20
        )),
        None => out.push(format!("the title's allocations from the process heap (XMemAllocDefault): {} MB in {}", title.0 >> 20, title.1)),
    }
    // (copied out under the lock, added up after it: adding up allocates)
    let mut alive: Vec<Site> = Vec::with_capacity(SLOTS);
    snapshot_into(&mut alive);
    // (xcl.exe's own base: in host mode the process's executable is the title's)
    let base = {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT};
        let mut m = std::ptr::null_mut();
        let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        unsafe { GetModuleHandleExW(flags, lines as *const u16, &mut m) };
        m as usize
    };
    for (large, what) in [(true, "large"), (false, "small")] {
        let mut by_site: std::collections::HashMap<[usize; FRAMES], (usize, usize)> = std::collections::HashMap::new();
        for s in alive.iter().filter(|s| (s.size >= LARGE) == large) {
            let e = by_site.entry(s.frames).or_default();
            e.0 += weight(s.size);
            e.1 += 1;
        }
        let mut sites: Vec<_> = by_site.into_iter().collect();
        sites.sort_by_key(|s| std::cmp::Reverse(s.1 .0));
        let total: usize = sites.iter().map(|s| s.1 .0).sum();
        out.push(if large {
            format!(
                "XCL's own heap, large allocations: {} MB in {} from {} places ({} not kept: table full); xcl.exe at {base:#x}",
                total >> 20,
                alive.iter().filter(|s| s.size >= LARGE).count(),
                sites.len(),
                UNTRACKED.load(Ordering::Relaxed)
            )
        } else {
            format!("XCL's own heap, small allocations: about {} MB, from 1 in {SAMPLE} of them kept ({} from {} places)", total >> 20, sites.iter().map(|s| s.1 .1).sum::<usize>(), sites.len())
        });
        for (frames, (bytes, n)) in sites.into_iter().take(top) {
            let stack: Vec<String> = frames.iter().take_while(|&&f| f != 0).map(|&f| if f >= base && f - base < (64 << 20) { format!("+{:#x}", f - base) } else { format!("{f:#x}") }).collect();
            if large {
                out.push(format!("XCL's own heap, large: {} MB in {} from {}", bytes >> 20, n, stack.join(" ")));
            } else {
                out.push(format!("XCL's own heap, small: about {} MB ({} kept) from {}", bytes >> 20, n, stack.join(" ")));
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn about_one_small_allocation_in_sample_is_kept() {
        let n = (0..1usize << 20).filter(|&k| kept(0x2_0000_0000 + 16 * k, 48)).count();
        let want = (1usize << 20) / SAMPLE;
        assert!(n > want / 2 && n < want * 2, "{n} of {} kept, want about {want}", 1 << 20);
        assert!(kept(0x1234_0000, LARGE), "every large one");
        assert_eq!(weight(LARGE), LARGE);
        assert_eq!(weight(100), 100 * SAMPLE);
    }

    #[test]
    fn the_reports_own_buffer_is_not_reported() {
        let mut alive: Vec<Site> = Vec::with_capacity(SLOTS);
        // (as the switch on would keep it: a large allocation of the report itself)
        added(alive.as_mut_ptr() as *mut u8, SLOTS * std::mem::size_of::<Site>());
        let other = 0x4321_0000usize;
        added(other as *mut u8, 2 << 20);
        snapshot_into(&mut alive);
        assert!(alive.iter().all(|s| s.ptr != alive.as_ptr() as usize), "the report's buffer is left out");
        assert!(alive.iter().any(|s| s.ptr == other), "an allocation of XCL's is kept");
        removed(other as *mut u8, 2 << 20);
        removed(alive.as_mut_ptr() as *mut u8, SLOTS * std::mem::size_of::<Site>());
    }

    #[test]
    fn sizes_fall_in_their_class() {
        assert_eq!(class(0), 0);
        assert_eq!(class(255), 0);
        assert_eq!(class(256), 1);
        assert_eq!(class((64 << 10) - 1), 2);
        assert_eq!(class(64 << 10), 3);
        assert_eq!(class(1 << 30), 6);
        assert_eq!(class(usize::MAX), 6);
    }

    #[test]
    fn a_large_allocation_is_kept_with_its_place_until_it_is_freed() {
        // (straight at the table: the switch is the process's, read once)
        let p = 0x1234_5670usize as *mut u8;
        added(p, 3 << 20);
        let found = locked(|t| t.iter().find(|s| s.ptr == p as usize).copied());
        let site = found.expect("kept");
        assert_eq!(site.size, 3 << 20);
        assert_ne!(site.frames[0], 0, "its call stack was taken");
        removed(p, 3 << 20);
        assert!(locked(|t| t.iter().all(|s| s.ptr != p as usize)));
        // a small one is kept only when its address is one of the sampled ones
        let mut picked = (0x1000_0000usize..).step_by(16).filter(|&a| kept(a, 100));
        let (q, r) = (picked.next().unwrap(), (0x1000_0000usize..).step_by(16).find(|&a| !kept(a, 100)).unwrap());
        added(q as *mut u8, 100);
        added(r as *mut u8, 100);
        assert!(locked(|t| t.iter().any(|s| s.ptr == q)), "a sampled small allocation is kept");
        assert!(locked(|t| t.iter().all(|s| s.ptr != r)), "another is only counted");
        removed(q as *mut u8, 100);
        removed(r as *mut u8, 100);
        assert!(locked(|t| t.iter().all(|s| s.ptr != q)));
    }
}
