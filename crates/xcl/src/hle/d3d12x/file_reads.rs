//! What file reads put into the title's graphics memory.
//!
//! An unbuffered read arrives by DMA: the CPU writes nothing, and the write watch reports at most
//! that the pages were locked for it – before the bytes are there. A look that takes that report in
//! between leaves the bytes out for good. So the range of a read into graphics memory is a write
//! XCL knows of (`title_textures::note_known_write`) once it is finished: at once when ReadFile says
//! so, else at the first look that finds its OVERLAPPED done.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const ERROR_IO_PENDING: u32 = 997;
const STATUS_PENDING: u64 = 0x103;
/// a read that looks unfinished for longer is taken to be done (its OVERLAPPED may be another's)
const GIVE_UP: Duration = Duration::from_secs(10);

struct Pending {
    overlapped: usize,
    at: usize,
    len: usize,
    since: Instant,
}

static PENDING: Mutex<Vec<Pending>> = Mutex::new(Vec::new());
static ANY: AtomicBool = AtomicBool::new(false);

/// ReadFile of `len` bytes to `at` returned `ok` (with `got`, and `error` when it did not).
pub unsafe fn returned(ok: bool, error: u32, at: *mut c_void, len: u32, got: *const u32, overlapped: *const c_void) {
    if at.is_null() || len == 0 {
        return;
    }
    let (a, n) = (at as usize, len as usize);
    let graphics = crate::gpu::uma::get().is_some_and(|u| u.contains(a)) || crate::hle::kernelx::in_graphics_reservation(a);
    if !graphics {
        return;
    }
    if ok {
        let got = if got.is_null() { n } else { (got.read_unaligned() as usize).min(n) };
        super::title_textures::note_known_write(a, got);
    } else if error == ERROR_IO_PENDING && !overlapped.is_null() {
        PENDING.lock().unwrap().push(Pending { overlapped: overlapped as usize, at: a, len: n, since: Instant::now() });
        ANY.store(true, Ordering::Release);
    }
}

/// The reads that finished since the last look are known writes.
pub unsafe fn take_finished() {
    if !ANY.load(Ordering::Acquire) {
        return;
    }
    let mut g = PENDING.lock().unwrap();
    g.retain(|p| {
        let pending = crate::gpu::shader::readable(p.overlapped, 8) && (p.overlapped as *const u64).read_volatile() == STATUS_PENDING;
        if pending && p.since.elapsed() < GIVE_UP {
            return true;
        }
        super::title_textures::note_known_write(p.at, p.len);
        false
    });
    ANY.store(!g.is_empty(), Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read is known when it returns, or – an overlapped one – at the first look that finds it
    /// done; reads elsewhere are nothing to XCL.
    #[test]
    fn a_finished_read_is_a_known_write() {
        let Some(u) = crate::gpu::uma::get() else { return };
        unsafe {
            let Some(p) = u.alloc(0x1_0000, 0x1000) else { return };
            let at = p as usize;
            let known = || super::super::title_textures::known_in(at, 0x1_0000);
            let mut heap = [0u8; 64];
            returned(true, 0, heap.as_mut_ptr() as *mut c_void, 64, std::ptr::null(), std::ptr::null());
            assert!(known().is_empty());
            let got = 0x300u32;
            returned(true, 0, p as *mut c_void, 0x1000, &got, std::ptr::null());
            assert_eq!(known(), vec![(at, 0x300)]);
            // an overlapped one, first unfinished
            let mut ov = [STATUS_PENDING, 0, 0, 0];
            returned(false, ERROR_IO_PENDING, (at + 0x4000) as *mut c_void, 0x2000, std::ptr::null(), ov.as_ptr() as *const c_void);
            returned(false, 5, (at + 0x8000) as *mut c_void, 0x2000, std::ptr::null(), ov.as_ptr() as *const c_void);
            take_finished();
            assert_eq!(known(), vec![(at, 0x300)], "still on its way");
            std::ptr::write_volatile(&mut ov[0], 0);
            take_finished();
            assert_eq!(known(), vec![(at, 0x300), (at + 0x4000, 0x2000)], "done (the refused one is not)");
            take_finished();
            assert_eq!(known().len(), 2, "once");
            super::super::title_textures::forget_known_in(at, 0x1_0000);
            u.free(at);
        }
    }
}
