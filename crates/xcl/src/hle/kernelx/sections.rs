//! Physical pages that show at every address they are mapped at.
//!
//! On the console a physical page mapped at several addresses is one page: a write through one
//! address is read at all of them. A retail title maps its streamed pages into a pool, into the
//! CPU view of a 24 GB reservation and, through its queue, into that reservation's GPU view.
//! Here a block of physical pages is one pagefile-backed section, and mapping a page puts a view
//! of its 64 KiB slice into a placeholder at the address: every view of the slice is the same
//! memory, with no copy. The section is `SEC_RESERVE`, so a page takes commit only once it is
//! committed through one of its views, and then once for all of them.
//!
//! Address space that may hold such views is reserved as a placeholder (`reserve`). A view
//! replaces the placeholder at its address (`map`); unmapping it leaves a placeholder again
//! (`unmap`), ready for the next view. Ordinary memory can replace part of a placeholder too
//! (`replace_with_reservation`), write-watched if asked – but `GetWriteWatch` refuses a range
//! that holds a view, so the watch must ask about such reservations one by one.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile3, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQuery, MEMORY_BASIC_INFORMATION,
    MEMORY_MAPPED_VIEW_ADDRESS, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_READWRITE, SEC_RESERVE, VIRTUAL_ALLOCATION_TYPE,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// One console physical page (64 KiB, the Windows allocation granularity too).
pub const PAGE: usize = 0x1_0000;

const MEM_WRITE_WATCH: VIRTUAL_ALLOCATION_TYPE = 0x20_0000;

/// A block of physical pages: one pagefile-backed section of `pages` 64 KiB pages.
pub struct Section {
    handle: HANDLE,
    pages: usize,
}

// SAFETY: a section handle may be used from any thread; the section itself is kernel state.
unsafe impl Send for Section {}
unsafe impl Sync for Section {}

impl Section {
    /// A section of `pages` pages, none committed. None if Windows refuses it.
    pub fn new(pages: usize) -> Option<Section> {
        let bytes = pages.checked_mul(PAGE).filter(|&b| b > 0)?;
        // SAFETY: a pagefile-backed mapping with no name and default security.
        let handle = unsafe { CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE, (bytes >> 32) as u32, bytes as u32, std::ptr::null()) };
        (!handle.is_null()).then_some(Section { handle, pages })
    }

    pub fn pages(&self) -> usize {
        self.pages
    }

    /// Map pages `first..first + n` of the section at `at`, which must be a placeholder or lie in
    /// one (`reserve`). The view is one Windows mapping: `unmap` takes all of it at once.
    ///
    /// # Safety
    /// `[at, at + n * PAGE)` must be address space the caller owns as a placeholder.
    pub unsafe fn map(&self, first: usize, n: usize, at: usize) -> bool {
        if n == 0 || first.checked_add(n).is_none_or(|end| end > self.pages) || !at.is_multiple_of(PAGE) {
            return false;
        }
        let len = n * PAGE;
        if !split(at, len) {
            return false;
        }
        let v = MapViewOfFile3(self.handle, GetCurrentProcess(), at as *const c_void, (first * PAGE) as u64, len, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, std::ptr::null_mut(), 0);
        v.Value as usize == at
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // views still mapped keep the section alive until they are unmapped
        // SAFETY: the handle is ours and closed once.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Reserve `len` bytes of placeholder address space, at `at` if given. None if it is taken.
///
/// # Safety
/// Only reserves address space; the caller releases it with `release`.
pub unsafe fn reserve(at: Option<usize>, len: usize) -> Option<usize> {
    if len == 0 || !len.is_multiple_of(PAGE) {
        return None;
    }
    let want = at.unwrap_or(0);
    let p = VirtualAlloc2(GetCurrentProcess(), want as *const c_void, len, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0) as usize;
    (p != 0 && (at.is_none() || p == want)).then_some(p)
}

/// The allocation `p` lies in: (base, size, state, type), from one VirtualQuery.
fn allocation_at(p: usize) -> Option<(usize, usize, u32, u32)> {
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: VirtualQuery only reads the address space description.
    if unsafe { VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) } == 0 {
        return None;
    }
    Some((m.AllocationBase as usize, m.RegionSize, m.State, m.Type))
}

/// Make `[at, at + len)` a placeholder of its own, splitting the placeholder it lies in. True if
/// it is one now (already exactly one, too).
///
/// # Safety
/// `[at, at + len)` must lie in one placeholder the caller owns.
pub unsafe fn split(at: usize, len: usize) -> bool {
    match allocation_at(at) {
        // a placeholder is a reservation (MEM_RESERVE) with no protection; one that is already
        // exactly this range cannot be split again – it is what was asked for
        Some((base, size, MEM_RESERVE, _)) if base == at && size == len => true,
        Some((_, _, MEM_RESERVE, _)) => VirtualFree(at as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0,
        _ => false,
    }
}

/// Unmap the view at `at` (all of it) and leave a placeholder in its place.
///
/// # Safety
/// `at` must be the start of a view made by `Section::map`.
pub unsafe fn unmap(at: usize) -> bool {
    UnmapViewOfFile2(GetCurrentProcess(), MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void }, MEM_PRESERVE_PLACEHOLDER) != 0
}

/// Commit `[at, at + len)` of a view: the section's pages there are committed for every view of
/// them. True if they are committed now.
///
/// # Safety
/// The range must lie in views made by `Section::map`.
pub unsafe fn commit(at: usize, len: usize) -> bool {
    !VirtualAlloc(at as *const c_void, len, MEM_COMMIT, PAGE_READWRITE).is_null()
}

/// Ordinary reserved memory in place of `[at, at + len)` of a placeholder, write-watched if
/// `write_watch` (commit it with `VirtualAlloc` as usual).
///
/// # Safety
/// `[at, at + len)` must lie in one placeholder the caller owns.
pub unsafe fn replace_with_reservation(at: usize, len: usize, write_watch: bool) -> bool {
    if !split(at, len) {
        return false;
    }
    let t = MEM_RESERVE | MEM_REPLACE_PLACEHOLDER | if write_watch { MEM_WRITE_WATCH } else { 0 };
    VirtualAlloc2(GetCurrentProcess(), at as *const c_void, len, t, PAGE_READWRITE, std::ptr::null_mut(), 0) as usize == at
}

/// Turn the reservation `replace_with_reservation` made at `[at, at + len)` back into a
/// placeholder (what was committed there is gone).
///
/// # Safety
/// `[at, at + len)` must be exactly such a reservation.
pub unsafe fn reservation_to_placeholder(at: usize, len: usize) -> bool {
    VirtualFree(at as *mut c_void, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0
}

/// Release every placeholder in `[at, at + len)`, which splitting may have cut into many (views
/// in it must be unmapped first). True if all of it went.
///
/// # Safety
/// `[at, at + len)` must be placeholders the caller owns.
pub unsafe fn release_all(at: usize, len: usize) -> bool {
    let mut p = at;
    while p < at + len {
        match allocation_at(p) {
            Some((base, size, MEM_RESERVE, _)) if base == p && release(p) => p += size,
            _ => return false,
        }
    }
    true
}

/// Release the placeholder at `at` (its whole extent; views in it must be unmapped first).
///
/// # Safety
/// `at` must be the start of a placeholder the caller owns.
pub unsafe fn release(at: usize) -> bool {
    VirtualFree(at as *mut c_void, 0, MEM_RELEASE) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{GetWriteWatch, MEM_MAPPED};

    /// Placeholder address space for a test, released at the end (views unmapped first by the test).
    struct Space(usize, usize);
    impl Space {
        fn new(pages: usize) -> Space {
            let len = pages * PAGE;
            Space(unsafe { reserve(None, len) }.expect("placeholder"), len)
        }
        fn at(&self, i: usize) -> usize {
            self.0 + i * PAGE
        }
    }
    impl Drop for Space {
        fn drop(&mut self) {
            // the test leaves placeholders behind; release each piece
            let mut p = self.0;
            while p < self.0 + self.1 {
                let (base, size, _, _) = allocation_at(p).unwrap();
                assert_eq!(base, p);
                unsafe { release(p) };
                p += size;
            }
        }
    }

    unsafe fn read(at: usize) -> u64 {
        std::ptr::read_volatile(at as *const u64)
    }

    unsafe fn write(at: usize, v: u64) {
        std::ptr::write_volatile(at as *mut u64, v)
    }

    #[test]
    fn a_page_mapped_at_two_addresses_is_one_page() {
        let s = Section::new(8).unwrap();
        let space = Space::new(16);
        unsafe {
            assert!(s.map(5, 1, space.at(2)));
            assert!(s.map(5, 1, space.at(11)));
            assert!(commit(space.at(2), PAGE));
            write(space.at(2) + 0x40, 0xDEAD_BEEF);
            assert_eq!(read(space.at(11) + 0x40), 0xDEAD_BEEF, "a write at one address is read at the other");
            write(space.at(11) + 0x40, 7);
            assert_eq!(read(space.at(2) + 0x40), 7);
            assert_eq!(allocation_at(space.at(11)).unwrap().3, MEM_MAPPED);
            assert!(unmap(space.at(2)) && unmap(space.at(11)));
        }
    }

    #[test]
    fn committing_through_one_view_commits_the_page_for_all() {
        let s = Section::new(2).unwrap();
        let space = Space::new(4);
        unsafe {
            assert!(s.map(1, 1, space.at(0)));
            assert!(s.map(1, 1, space.at(3)));
            assert_eq!(allocation_at(space.at(3)).unwrap().2, MEM_RESERVE, "a page is not committed before it is used");
            assert!(commit(space.at(0), PAGE));
            assert_eq!(allocation_at(space.at(3)).unwrap().2, MEM_COMMIT);
            assert_eq!(read(space.at(3)), 0, "a new page reads zeros");
            assert!(unmap(space.at(0)) && unmap(space.at(3)));
        }
    }

    #[test]
    fn unmapping_leaves_a_placeholder_for_the_next_page() {
        let s = Section::new(4).unwrap();
        let space = Space::new(2);
        unsafe {
            assert!(s.map(0, 1, space.at(1)) && commit(space.at(1), PAGE));
            write(space.at(1), 1);
            assert!(unmap(space.at(1)));
            let (base, size, state, _) = allocation_at(space.at(1)).unwrap();
            assert_eq!((base, size, state), (space.at(1), PAGE, MEM_RESERVE), "a placeholder of the page's size is left");
            assert!(s.map(3, 1, space.at(1)) && commit(space.at(1), PAGE));
            assert_eq!(read(space.at(1)), 0, "the other page is there now");
            // the first page kept its contents while it was mapped nowhere
            assert!(s.map(0, 1, space.at(0)));
            assert_eq!(read(space.at(0)), 1);
            assert!(unmap(space.at(0)) && unmap(space.at(1)));
        }
    }

    #[test]
    fn a_run_of_pages_is_one_view() {
        let s = Section::new(8).unwrap();
        let space = Space::new(8);
        unsafe {
            assert!(s.map(2, 4, space.at(1)));
            assert_eq!(allocation_at(space.at(1)).unwrap().0, space.at(1));
            assert_eq!(allocation_at(space.at(4)).unwrap().0, space.at(1), "one view covers the run");
            assert!(commit(space.at(1), 4 * PAGE));
            assert!(s.map(4, 1, space.at(7)));
            write(space.at(3), 42); // page 4 of the section, third of the run
            assert_eq!(read(space.at(7)), 42);
            assert!(!s.map(6, 4, space.at(0)), "pages past the end of the section are refused");
            assert!(unmap(space.at(1)) && unmap(space.at(7)));
        }
    }

    #[test]
    fn write_watched_memory_lives_beside_views_in_one_placeholder() {
        let s = Section::new(1).unwrap();
        let space = Space::new(8);
        unsafe {
            assert!(replace_with_reservation(space.at(0), 4 * PAGE, true));
            assert!(!VirtualAlloc(space.at(0) as *const c_void, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(s.map(0, 1, space.at(6)) && commit(space.at(6), PAGE));
            write(space.at(1), 1);
            write(space.at(6), 2);
            let mut out = [std::ptr::null_mut::<c_void>(); 64];
            let (mut n, mut gran) = (out.len(), 0u32);
            assert_eq!(GetWriteWatch(1, space.at(0) as *const c_void, 4 * PAGE, out.as_mut_ptr(), &mut n, &mut gran), 0);
            assert_eq!((n, out[0] as usize), (1, space.at(1)), "the write-watched reservation reports its write");
            let mut n = out.len();
            assert_ne!(GetWriteWatch(0, space.at(0) as *const c_void, 8 * PAGE, out.as_mut_ptr(), &mut n, &mut gran), 0, "a range with a view in it is refused");
            assert!(reservation_to_placeholder(space.at(0), 4 * PAGE));
            assert_eq!(allocation_at(space.at(0)).unwrap().2, MEM_RESERVE);
            assert!(unmap(space.at(6)));
        }
    }

    #[test]
    fn splitting_a_range_that_is_already_a_placeholder_is_a_no_op() {
        let space = Space::new(3);
        unsafe {
            assert!(split(space.at(1), PAGE));
            assert!(split(space.at(1), PAGE));
            assert_eq!(allocation_at(space.at(1)).unwrap().0, space.at(1));
            assert_eq!(allocation_at(space.at(2)).unwrap().0, space.at(2));
        }
    }

    #[test]
    fn a_section_of_no_pages_is_refused() {
        assert!(Section::new(0).is_none());
    }
}
