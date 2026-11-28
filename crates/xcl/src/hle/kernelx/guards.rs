//! CPU writes to the views of physical pages, caught page by page.
//!
//! The upload of title memory to video memory (`d3d12x::sparse`) learns what the CPU wrote from
//! the write watch, and a view of a section (`sections`) cannot be write-watched. So the views
//! of a physical page are kept read-only: the first write through any of them faults, the
//! handler makes that view writable again and notes the page, and the write goes on. Before the
//! title's next submit `take_dirty` hands over the pages of every view of each page written –
//! a write through one view is a write to all of them – and guards them again.
//!
//! A kernel call does not fault into the handler: it fails. Memory a kernel call or XCL is about
//! to write goes through `lazy::ensure`, which opens the guards there first (`open`).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{VirtualProtect, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

#[derive(Default)]
struct State {
    /// view page (64 KiB aligned) → the physical page it shows
    at: HashMap<usize, usize>,
    /// physical page → the view pages that show it
    views: HashMap<usize, Vec<usize>>,
    /// physical pages written since the last `take_dirty`
    dirty: HashSet<usize>,
    /// views opened for a write still going on (`Opened`): not guarded again until it ends
    pinned: HashMap<usize, usize>,
    /// overlapped reads into views: (the OVERLAPPED, its views kept open until it is done)
    pending: Vec<(usize, Opened)>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Whether any view was ever watched: `open` and `committed` sit on the file-read and
/// DirectStorage paths and take no lock before then.
static WATCHING: AtomicBool = AtomicBool::new(false);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(State::default))
}

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

unsafe fn protect(view: usize, p: u32) -> bool {
    let mut old = 0;
    VirtualProtect(view as *const c_void, PAGE, p, &mut old) != 0
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    // a write (access 1) to a guarded view
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 || r.ExceptionInformation[0] != 1 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let view = r.ExceptionInformation[1] & !(PAGE - 1);
    // (a view not committed yet cannot be opened: that fault is `lazy`'s, which commits the page
    // and counts it as written – claimed here, the write would fault again for ever)
    let ours = with(|s| match s.at.get(&view) {
        Some(&page) if unsafe { protect(view, PAGE_READWRITE) } => {
            s.dirty.insert(page);
            true
        }
        _ => false,
    });
    // (another thread may have opened the view between the fault and here: writable now)
    if ours || writable(view) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

/// Whether `view` is a guarded view that is writable now.
fn writable(view: usize) -> bool {
    if !with(|s| s.at.contains_key(&view)) {
        return false;
    }
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: VirtualQuery only reads the address space description.
    let n = unsafe { VirtualQuery(view as *const c_void, &mut m, std::mem::size_of_val(&m)) };
    n != 0 && m.State == MEM_COMMIT && m.Protect == PAGE_READWRITE
}

/// Physical page `page` is shown at `view` (64 KiB aligned): writes there are caught from now on.
/// A view not committed yet cannot be guarded; `committed` notes it when it is.
pub fn watch(page: usize, view: usize) {
    watch_with(page, view, true);
}

/// Like `watch`, but skips the `VirtualProtect` syscall when `already_committed` is false
/// (an uncommitted view cannot be protected yet and will be marked dirty when lazily committed).
pub fn watch_with(page: usize, view: usize, already_committed: bool) {
    install();
    WATCHING.store(true, Ordering::Release);
    with(|s| {
        if let Some(old) = s.at.insert(view, page).filter(|&old| old != page) {
            if let Some(v) = s.views.get_mut(&old) {
                v.retain(|&a| a != view);
            }
        }
        let v = s.views.entry(page).or_default();
        if !v.contains(&view) {
            v.push(view);
        }
        if already_committed {
            unsafe { protect(view, PAGE_READONLY) };
        }
    });
}

/// `view` no longer shows a physical page (it was unmapped): its writes are not looked for.
pub fn forget(view: usize) {
    with(|s| {
        if let Some(page) = s.at.remove(&view) {
            if let Some(v) = s.views.get_mut(&page) {
                v.retain(|&a| a != view);
                if v.is_empty() {
                    s.views.remove(&page);
                    s.dirty.remove(&page);
                }
            }
        }
    });
}

/// `view` was committed just now (it reads zeros and is writable): its page counts as written.
pub fn committed(view: usize) {
    if !WATCHING.load(Ordering::Acquire) {
        return;
    }
    with(|s| {
        if let Some(&page) = s.at.get(&(view & !(PAGE - 1))) {
            s.dirty.insert(page);
        }
    });
}

/// Physical page `page` was written other than through a view (XCL by page number, or the page
/// came back from an address outside the views): every view of it counts as written.
pub fn written(page: usize) {
    if !WATCHING.load(Ordering::Acquire) {
        return;
    }
    with(|s| {
        if s.views.contains_key(&page) {
            s.dirty.insert(page);
        }
    });
}

/// Views opened by `open` for a write: they stay writable until this is dropped, when the write
/// is over and their pages count as written again.
#[must_use = "the views stay open only while this lives"]
pub struct Opened(Vec<usize>);

impl Drop for Opened {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        with(|s| {
            for view in self.0.drain(..) {
                if let Some(n) = s.pinned.get_mut(&view) {
                    *n -= 1;
                    if *n == 0 {
                        s.pinned.remove(&view);
                    }
                }
                if let Some(&page) = s.at.get(&view) {
                    s.dirty.insert(page);
                }
            }
        });
    }
}

/// `[p, p + len)` is about to be written by a kernel call or XCL: the views there are made
/// writable, stay so while the returned `Opened` lives (a kernel call fails on a view guarded
/// again under it), and their pages count as written.
pub fn open(p: usize, len: usize) -> Opened {
    if len == 0 || !WATCHING.load(Ordering::Acquire) {
        return Opened(Vec::new());
    }
    with(|s| {
        let mut opened = Vec::new();
        if s.at.is_empty() {
            return Opened(opened);
        }
        let mut view = p & !(PAGE - 1);
        while view < p.saturating_add(len) {
            if let Some(&page) = s.at.get(&view) {
                s.dirty.insert(page);
                unsafe { protect(view, PAGE_READWRITE) };
                *s.pinned.entry(view).or_default() += 1;
                opened.push(view);
            }
            view += PAGE;
        }
        Opened(opened)
    })
}

/// An overlapped read (`ov`) writes into the views `opened` holds after its call returned: they
/// stay open until it is done (`take_dirty` looks).
pub fn open_until_done(opened: Opened, ov: usize) {
    if !opened.0.is_empty() {
        with(|s| s.pending.push((ov, opened)));
    }
}

/// Whether the overlapped I/O at `ov` is still going on (its `Internal` is STATUS_PENDING). The
/// title may have freed the OVERLAPPED once it is done, so it is read without faulting.
fn still_pending(ov: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const STATUS_PENDING: usize = 0x103;
    let mut internal = 0usize;
    let mut n = 0usize;
    // SAFETY: ReadProcessMemory on our own process fails instead of faulting on an unreadable address.
    let ok = unsafe { ReadProcessMemory(GetCurrentProcess(), ov as *const c_void, &mut internal as *mut usize as *mut c_void, std::mem::size_of::<usize>(), &mut n) } != 0;
    ok && n == std::mem::size_of::<usize>() && internal == STATUS_PENDING
}

/// The pages written since the last call, as the `step`-sized pages (4 KiB for the write watch)
/// of every view of each, appended to `out`; those views are guarded again.
pub(super) fn take_dirty(out: &mut Vec<usize>, step: usize) {
    collect_dirty(out, step, &mut Vec::new());
}

/// Also mark the canonical storage used for copies to GPU-view tiles.
pub(crate) fn take_uploads(out: &mut Vec<usize>, step: usize) {
    let mut pages = Vec::new();
    collect_dirty(out, step, &mut pages);
    for page in pages {
        if let Some((at, true)) = super::alias::backing(page) {
            out.extend((at..at + PAGE).step_by(step));
        }
    }
}

fn collect_dirty(out: &mut Vec<usize>, step: usize, pages: &mut Vec<usize>) {
    if !WATCHING.load(Ordering::Acquire) {
        return;
    }
    // overlapped reads that are done let their views go (dropped outside the lock: it takes it)
    let done: Vec<(usize, Opened)> = with(|s| {
        let (done, pending) = std::mem::take(&mut s.pending).into_iter().partition(|&(ov, _)| !still_pending(ov));
        s.pending = pending;
        done
    });
    drop(done);
    with(|s| {
        for page in std::mem::take(&mut s.dirty) {
            pages.push(page);
            for &view in s.views.get(&page).map(Vec::as_slice).unwrap_or_default() {
                // guarded before its bytes are read for the upload: a write from here on is
                // caught again, one before this is in what the upload reads. A view opened for a
                // write still going on stays open (`Opened` counts its page as written at the end)
                if !s.pinned.contains_key(&view) {
                    unsafe { protect(view, PAGE_READONLY) };
                }
                out.extend((view..view + PAGE).step_by(step));
            }
        }
    });
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::sections::{commit, reserve, unmap, Section};
    use super::*;
    use std::sync::Mutex as TestLock;

    /// the guard state is one per process: tests that use it (here and in `alias`) run one at a time
    pub(in crate::hle::kernelx) static SERIAL: TestLock<()> = TestLock::new(());

    /// A section of `pages` pages and placeholder space for `views` views; page `i` of the
    /// section at view slot `j` is `space + j * PAGE`.
    struct Fixture {
        section: Section,
        space: usize,
        mapped: Vec<usize>,
    }
    impl Fixture {
        fn new(pages: usize, views: usize) -> Fixture {
            Fixture { section: Section::new(pages).unwrap(), space: unsafe { reserve(None, views * PAGE) }.unwrap(), mapped: Vec::new() }
        }
        /// Section page `page` at view slot `slot`, committed and guarded; its physical page id is `page`.
        fn show(&mut self, page: usize, slot: usize) -> usize {
            let at = self.space + slot * PAGE;
            unsafe {
                assert!(self.section.map(page, 1, at));
                assert!(commit(at, PAGE));
            }
            watch(page, at);
            self.mapped.push(at);
            at
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            for &at in &self.mapped {
                forget(at);
                unsafe { unmap(at) };
            }
        }
    }

    fn dirty() -> Vec<usize> {
        let mut out = Vec::new();
        take_dirty(&mut out, PAGE);
        out.sort_unstable();
        out
    }

    unsafe fn write(at: usize, v: u64) {
        std::ptr::write_volatile(at as *mut u64, v)
    }

    #[test]
    fn a_write_through_one_view_reports_every_view_of_the_page() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(4, 8);
        let a = f.show(2, 1);
        let b = f.show(2, 5);
        let c = f.show(3, 6);
        dirty();
        unsafe { write(b + 16, 9) };
        assert_eq!(unsafe { std::ptr::read_volatile((a + 16) as *const u64) }, 9);
        assert_eq!(dirty(), vec![a, b], "both views of the written page, not the other page's");
        assert_eq!(dirty(), Vec::<usize>::new(), "nothing written since");
        unsafe { write(a, 1) };
        unsafe { write(c, 1) };
        assert_eq!(dirty(), vec![a, b, c], "the views are guarded again after a take");
    }

    #[test]
    fn steps_of_4_kib_cover_the_view() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(1, 1);
        let a = f.show(0, 0);
        dirty();
        unsafe { write(a + 0x8000, 1) };
        let mut out = Vec::new();
        take_dirty(&mut out, 0x1000);
        assert_eq!(out, (a..a + PAGE).step_by(0x1000).collect::<Vec<_>>());
    }

    #[test]
    fn open_makes_views_writable_for_a_kernel_call_and_counts_the_write() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(2, 4);
        let a = f.show(0, 0);
        let b = f.show(1, 1);
        dirty();
        let opened = open(a + PAGE - 8, 16); // straddles both views
        assert!(writable(a) && writable(b));
        assert_eq!(dirty(), vec![a, b]);
        assert!(writable(a) && writable(b), "a take leaves views open while their write goes on");
        drop(opened);
        assert_eq!(dirty(), vec![a, b], "the end of the write counts as a write");
        assert!(!writable(a), "guarded again after the take that follows");
    }

    #[test]
    fn views_an_overlapped_read_writes_stay_open_until_it_is_done() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(1, 1);
        let a = f.show(0, 0);
        dirty();
        // an OVERLAPPED whose Internal says pending, then done
        let mut ov = [0x103usize, 0, 0, 0];
        open_until_done(open(a, 16), ov.as_mut_ptr() as usize);
        assert_eq!(dirty(), vec![a]);
        assert!(writable(a), "still open while the read is pending");
        ov[0] = 0;
        assert_eq!(dirty(), vec![a], "the end of the read counts as a write");
        assert!(!writable(a), "guarded again once the read is done");
    }

    #[test]
    fn a_forgotten_view_is_no_longer_reported() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(1, 2);
        let a = f.show(0, 0);
        let b = f.show(0, 1);
        dirty();
        forget(b);
        unsafe { write(a, 3) };
        assert_eq!(dirty(), vec![a]);
    }

    #[test]
    fn writes_from_many_threads_are_all_caught_and_kept() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        const PAGES: usize = 64;
        const THREADS: usize = 8;
        let mut f = Fixture::new(PAGES, PAGES);
        let views: Vec<usize> = (0..PAGES).map(|i| f.show(i, i)).collect();
        dirty();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let (views, barrier) = (views.clone(), barrier.clone());
                std::thread::spawn(move || {
                    barrier.wait();
                    for (i, &v) in views.iter().enumerate() {
                        unsafe { write(v + t * 64, (i * THREADS + t) as u64) };
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        for (i, &v) in views.iter().enumerate() {
            for t in 0..THREADS {
                assert_eq!(unsafe { std::ptr::read_volatile((v + t * 64) as *const u64) }, (i * THREADS + t) as u64);
            }
        }
        assert_eq!(dirty(), views);
    }

    /// A write to a guarded view that is not committed yet is not the guards' to handle (the
    /// lazy handler commits it): claimed, it faulted again for ever whenever this handler ran
    /// before the lazy one.
    #[test]
    fn a_write_to_a_view_not_committed_yet_is_left_to_the_next_handler() {
        use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, EXCEPTION_RECORD};
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Fixture::new(2, 2);
        // page 0 mapped but never committed; page 1 committed and guarded
        let fresh = f.space;
        unsafe { assert!(f.section.map(0, 1, fresh)) };
        watch(0, fresh);
        f.mapped.push(fresh);
        let used = f.show(1, 1);
        let fault = |at: usize| unsafe {
            let mut rec: EXCEPTION_RECORD = std::mem::zeroed();
            rec.ExceptionCode = EXCEPTION_ACCESS_VIOLATION as NTSTATUS;
            rec.NumberParameters = 2;
            rec.ExceptionInformation[0] = 1;
            rec.ExceptionInformation[1] = at + 8;
            let mut ctx: CONTEXT = std::mem::zeroed();
            let mut ptrs = EXCEPTION_POINTERS { ExceptionRecord: &mut rec, ContextRecord: &mut ctx };
            handler(&mut ptrs)
        };
        assert_eq!(fault(fresh), EXCEPTION_CONTINUE_SEARCH, "not committed: left to the lazy handler");
        assert_eq!(fault(used), EXCEPTION_CONTINUE_EXECUTION, "committed and guarded: opened here");
    }

    #[test]
    fn a_view_committed_later_counts_as_written() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let f = Fixture::new(1, 1);
        let at = f.space;
        unsafe { assert!(f.section.map(0, 1, at)) };
        watch(0, at); // not committed: cannot be guarded yet
        dirty();
        unsafe { assert!(commit(at, PAGE)) };
        committed(at);
        assert_eq!(dirty(), vec![at]);
        forget(at);
        unsafe { unmap(at) };
    }
}
