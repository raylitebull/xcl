//! What a command list is given in a GPU view of the title's memory (`XCL_HEAP_WINDOWS=1`,
//! `XCL_UMA_WINDOWS=16`: the view lies in the view windows): an index or vertex buffer there goes
//! to the PC at its view window's address and never as the console address, which is no PC
//! address; a value there (a wait's, a sync cell's) is the CPU view's. Its own process: the window
//! layout is chosen at the first `uma::get()`, from the environment. No device is needed (without
//! one no window has a PC address yet).

use xcl::gpu::uma;
use xcl::hle::d3d12x::list_x;
use xcl::hle::kernelx;

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
/// MEM_LARGE_PAGES | MEM_PHYSICAL: the console's 64 KiB pages
const MEM_64K_PAGES: u32 = 0x2040_0000;
const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_GRAPHICS_NOACCESS: u32 = 0x800;

/// the title's views: 24 GiB each (as a retail title reserves them: the CPU view stays out of the windows)
const SIZE: usize = 24 << 30;

#[test]
fn a_list_takes_a_gpu_view_address_to_its_window_and_its_values_to_the_cpu_view() {
    std::env::set_var("XCL_HEAP_WINDOWS", "1");
    std::env::set_var("XCL_UMA_WINDOWS", "16");
    std::env::set_var("XCL_COHERENT_MB", "512");
    let u = uma::get().expect("unified memory");
    assert_eq!(u.shader_windows(), uma::MAX_HEAP_WINDOWS, "the 16-window layout was not reserved");
    let null = std::ptr::null();
    unsafe {
        let c = kernelx::VirtualAlloc(null, SIZE, MEM_RESERVE | MEM_64K_PAGES, PAGE_READWRITE) as usize;
        assert_ne!(c, 0, "the CPU view");
        let g = kernelx::VirtualAlloc(null, SIZE, MEM_RESERVE | MEM_64K_PAGES, PAGE_NOACCESS | PAGE_GRAPHICS_NOACCESS) as usize;
        assert_ne!(g, 0, "the GPU view");
        assert!(u.is_view(g), "the GPU view is in a view window");
        assert_eq!(kernelx::VirtualAlloc(c as *const _, 0x1_0000, MEM_COMMIT, PAGE_READWRITE) as usize, c);

        for x in [0usize, 0x40, 0x1_0008, SIZE - 8] {
            // no window has a PC address without a device: none, never the console address
            assert_eq!(list_x::pc_va_of((g + x) as u64), 0, "G + {x:#x}");
            assert_eq!(list_x::cpu_at((g + x) as u64), (c + x) as u64, "G + {x:#x}");
            assert_eq!(list_x::cpu_at((c + x) as u64), (c + x) as u64, "C + {x:#x}");
        }
        // the CPU view is no unified memory: as it is
        assert_eq!(list_x::pc_va_of(c as u64), c as u64);

        assert_eq!(kernelx::VirtualFree(g as *mut _, 0, MEM_RELEASE), 1);
        assert_eq!(list_x::cpu_at(g as u64), g as u64, "unpaired: no CPU view");
        assert_eq!(kernelx::VirtualFree(c as *mut _, 0, MEM_RELEASE), 1);
    }
}
