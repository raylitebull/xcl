//! A retail title reserves 24 GiB of 64 KiB pages twice: a CPU view (read/write, it holds the
//! bytes) and a GPU-only view of the same pages, whose addresses are all the GPU is given. With
//! 16 memory windows (`XCL_HEAP_WINDOWS=1`, `XCL_UMA_WINDOWS=16`) the GPU view goes to the view
//! windows, where shaders reach it, and still names its CPU view; the CPU view stays plain
//! write-watched memory outside the windows. Its own process: the window layout is chosen at the
//! first `uma::get()`, from the environment. No device is needed.

use std::ffi::c_void;

use xcl::gpu::uma;
use xcl::hle::kernelx::{self, mirrors};

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_DECOMMIT: u32 = 0x4000;
const MEM_RELEASE: u32 = 0x8000;
/// MEM_LARGE_PAGES | MEM_PHYSICAL: the console's 64 KiB pages
const MEM_64K_PAGES: u32 = 0x2040_0000;
const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_GRAPHICS_NOACCESS: u32 = 0x800;

/// the title's views: 24 GiB each
const SIZE: usize = 24 << 30;

#[test]
fn a_gpu_view_lands_in_a_view_window_and_names_its_cpu_view() {
    std::env::set_var("XCL_HEAP_WINDOWS", "1");
    std::env::set_var("XCL_UMA_WINDOWS", "16");
    // (the coherent window as small as it gets: not the size an earlier run of a title noted)
    std::env::set_var("XCL_COHERENT_MB", "512");
    let u = uma::get().expect("unified memory");
    assert_eq!(u.shader_windows(), uma::MAX_HEAP_WINDOWS, "the 16-window layout was not reserved");
    let aperture = u.base()..u.base() + uma::MAX_HEAP_WINDOWS * uma::WINDOW;
    let null = std::ptr::null();
    unsafe {
        let c = kernelx::VirtualAlloc(null, SIZE, MEM_RESERVE | MEM_64K_PAGES, PAGE_READWRITE) as usize;
        assert_ne!(c, 0, "the CPU view");
        let g = kernelx::VirtualAlloc(null, SIZE, MEM_RESERVE | MEM_64K_PAGES, PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS) as usize;
        assert_ne!(g, 0, "the GPU view");

        // G: in the view windows, from a window's start, all of it
        let k = u.window_of(g).expect("the GPU view is in a window");
        assert!(u.is_view_window(k), "window {k} is no view window");
        assert_eq!(g, u.window(k).0);
        assert!(u.is_view(g + SIZE - 1) && u.is_title_address(g) && u.is_title_address(g + SIZE - 1));
        assert!(!u.contains(g) && !u.is_sparse(g), "the CPU cannot touch a GPU view");
        for x in [0, 0x1234_5678, 5 << 30, SIZE - 0x1_0000] {
            assert_eq!(mirrors::source(g + x), Some(c + x), "G + {x:#x}");
        }

        // C: not in the windows (never sparse video memory), plain write-watched memory the CPU writes
        assert!(!aperture.contains(&c) && !aperture.contains(&(c + SIZE - 1)), "the CPU view {c:#x} is in the aperture {aperture:x?}");
        assert!(u.window_of(c).is_none() && !u.contains(c) && !u.is_title_address(c));
        assert!(kernelx::in_graphics_reservation(c));
        let page = kernelx::VirtualAlloc(c as *const c_void, 0x1_0000, MEM_COMMIT, PAGE_READWRITE) as usize;
        assert_eq!(page, c);
        std::ptr::write_volatile((c + 8) as *mut u64, 0x5843_4C00_0000_0016);

        // the title commits at G and decommits it again: plain pages, nothing breaks, the pair stays
        assert_ne!(kernelx::VirtualAlloc((g + 0x2_0000) as *const c_void, 0x1_0000, MEM_COMMIT, PAGE_READWRITE), std::ptr::null_mut());
        assert_eq!(kernelx::VirtualFree((g + 0x2_0000) as *mut c_void, 0x1_0000, MEM_DECOMMIT), 1);
        assert_eq!(mirrors::source(g + 0x2_0000), Some(c + 0x2_0000));
        // a release names the reservation's start, as Windows wants it
        assert_eq!(kernelx::VirtualFree((g + 0x1_0000) as *mut c_void, 0, MEM_RELEASE), 0);

        // released: unpaired, and its range is free for the next GPU view
        assert_eq!(kernelx::VirtualFree(g as *mut c_void, 0, MEM_RELEASE), 1);
        assert_eq!(mirrors::source(g), None);
        assert!(!u.is_title_address(g) && u.view_reservation(g).is_none());
        let again = kernelx::VirtualAlloc(null, SIZE, MEM_RESERVE | MEM_64K_PAGES, PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS) as usize;
        assert_eq!(again, g, "the view windows' range came back whole");
        assert_eq!(mirrors::source(again + 0x10_0000), Some(c + 0x10_0000));
        assert_eq!(std::ptr::read_volatile((c + 8) as *const u64), 0x5843_4C00_0000_0016);

        // the CPU view goes, and the GPU view no longer names it
        assert_eq!(kernelx::VirtualFree(c as *mut c_void, 0, MEM_RELEASE), 1);
        assert_eq!(mirrors::source(again), None);
        assert!(!kernelx::in_graphics_reservation(c));
        assert_eq!(kernelx::VirtualFree(again as *mut c_void, 0, MEM_RELEASE), 1);
        assert!(u.view_reservation(again).is_none());
    }
}
