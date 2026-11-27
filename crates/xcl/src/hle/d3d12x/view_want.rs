//! The title's GPU view of its own memory, as the GPU is told about it.
//!
//! A retail title reserves two ranges for the same physical pages: one the CPU reads and writes,
//! and a GPU view the CPU cannot touch (`kernelx::mirrors`). It gives the GPU only addresses in
//! the GPU view: buffer views made by address, root views, vertex and index buffers, the
//! operands of memory operations. With the 16-window layout that view lies in view windows
//! (`Uma::is_view`), whose 64 KiB tiles get video memory, filled from the CPU's range, when an
//! address in them is first used (`sparse::map_mirror`).
//!
//! Every path that takes such an address from the title calls `want` with what it knows of the
//! range. The counters say which paths a title uses and how much of the view each asks for.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// The paths a GPU-view address reaches the GPU by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// a buffer view made by address (device #70 / #71)
    ViewAt = 0,
    /// a constant buffer view
    Cbv,
    /// a view of a buffer resource placed in the view
    BufferView,
    /// a buffer resource placed in the view
    PlacedBuffer,
    /// a root CBV / SRV / UAV (its size is not known)
    RootView,
    /// a vertex or index buffer
    VertexIndex,
    /// a memory operation of the list (copy, fill, write, timestamps, predication)
    MemoryOp,
    /// a wait on a value in memory
    Wait,
    /// a file read into the view
    FileRead,
    /// a Map of a resource in the view
    MapRead,
    /// a GPU copy or query resolve into a buffer in the view
    CopyInto,
    /// a root view of a two-pass mesh stage
    MeshStage,
    /// an acceleration-structure view (asked for from its root on: its size is not known)
    Bvh,
}

const PATHS: usize = 13;
const NAMES: [&str; PATHS] = [
    "views by address",
    "constant buffer views",
    "buffer views",
    "placed buffers",
    "root views",
    "vertex / index buffers",
    "memory operations",
    "waits",
    "file reads",
    "maps",
    "copies into it",
    "mesh stage views",
    "acceleration-structure views",
];

#[allow(clippy::declare_interior_mutable_const)]
const ZERO: AtomicU64 = AtomicU64::new(0);
static CALLS: [AtomicU64; PATHS] = [ZERO; PATHS];
static BYTES: [AtomicU64; PATHS] = [ZERO; PATHS];

/// Tile spans a thread remembers having asked for (`asked_lately`).
const LATELY: usize = 32;
/// The tiles of a view window (64 KiB, `sparse::map_mirror`)
const TILE_SHIFT: u32 = 16;
/// An empty place of `ASKED`: a span no other is inside
const NO_SPAN: (u64, u64) = (u64::MAX, 0);

thread_local! {
    /// The tile spans (first and last 64 KiB tile) this thread asked for last, the generation
    /// they were asked in (`generation`: of the GPU views and of the asks forgotten since) and
    /// where the next one goes. A title binds the same vertex buffers and root views for draw after draw, on the
    /// thread that records the list, and root views at a new offset of the same tiles (a
    /// constant allocator moving on for each draw); asking again walks every tile of the range
    /// (`sparse::map_mirror`: a shared set).
    static ASKED: Cell<([(u64, u64); LATELY], (u64, u64), usize)> = const { Cell::new(([NO_SPAN; LATELY], (0, 0), 0)) };
}

/// What an ask is remembered under: the GPU views as they are (`kernelx::gpu_view_map_gen`) and
/// the asks `sparse` has forgotten so far (`sparse::mirror_asks_forgotten`). A tile given back
/// when the title decommits its source may be asked for again, and must be when the title
/// commits and binds that memory again: a thread that still remembered asking for it left the
/// tile unmapped, and the draw read zeros through the view (a mesh without vertices).
pub(super) fn generation() -> (u64, u64) {
    (crate::hle::kernelx::gpu_view_map_gen(), super::sparse::mirror_asks_forgotten())
}

/// Whether every tile of `[va, va + len)` lies in one of the tile spans this thread asked for
/// last (`LATELY` of them) in this `generation`; its span is remembered if not. A view that went
/// or came since, or an ask forgotten since, empties the memory. A range reaching one tile past
/// what was asked for is not in it.
fn asked_lately(va: u64, len: u64, generation: (u64, u64)) -> bool {
    let span = (va >> TILE_SHIFT, va.saturating_add(len.saturating_sub(1)) >> TILE_SHIFT);
    ASKED.with(|a| {
        let (mut spans, asked_in, mut next) = a.get();
        if asked_in != generation {
            (spans, next) = ([NO_SPAN; LATELY], 0);
        } else if spans.iter().any(|&(first, last)| first <= span.0 && span.1 <= last) {
            return true;
        }
        spans[next] = span;
        a.set((spans, generation, (next + 1) % LATELY));
        false
    })
}

/// Whether `va` lies in a view window (false without the 16-window layout).
pub fn is_view(va: u64) -> bool {
    crate::gpu::uma::get().is_some_and(|u| u.is_view(va as usize))
}

/// The title gave `[va, va + len)` to the GPU on `path`: when it lies in a view window its
/// tiles are asked for (mapped and filled from the CPU's range at the next submit). True when
/// it was a view address.
pub fn want(path: Path, va: u64, len: u64) -> bool {
    if va == 0 || len == 0 || !is_view(va) {
        return false;
    }
    CALLS[path as usize].fetch_add(1, Ordering::Relaxed);
    let len = reach(path, va, len);
    if !asked_lately(va, len, generation()) {
        let queued = super::sparse::map_mirror(va as usize, len as usize) as u64;
        if queued != 0 {
            BYTES[path as usize].fetch_add(queued << TILE_SHIFT, Ordering::Relaxed);
        }
    }
    true
}

/// `XCL_VIEW_REACH_MB=n` (an experiment): a view whose size the title does not give – a root
/// view, a mesh stage's root view, an acceleration structure – is asked for `n` MiB from its
/// address on instead of 64 KiB, as far as that still lies in a view window. Tiles whose CPU
/// pages are not committed get no video memory (`sparse::fill_later`), so the cost is what the
/// title really has there. Off (0) by default.
fn reach(path: Path, va: u64, len: u64) -> u64 {
    static MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let mb = *MB.get_or_init(|| {
        let mb = std::env::var("XCL_VIEW_REACH_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0).min(4096);
        if mb != 0 {
            crate::xlog_info!("D3D12.X: XCL_VIEW_REACH_MB: root views, mesh stage views and acceleration structures are asked for {mb} MiB from their address on");
        }
        mb
    });
    // `XCL_VIEW_REACH=bvh,root,mesh`: only those kinds (all three without it)
    static KINDS: std::sync::OnceLock<[bool; 3]> = std::sync::OnceLock::new();
    let kinds = KINDS.get_or_init(|| match std::env::var("XCL_VIEW_REACH") {
        Ok(v) => ["bvh", "root", "mesh"].map(|k| v.split(',').any(|x| x.trim().eq_ignore_ascii_case(k))),
        Err(_) => [true; 3],
    });
    let sizeless = match path {
        Path::Bvh => kinds[0],
        Path::RootView => kinds[1],
        Path::MeshStage => kinds[2],
        _ => false,
    };
    longer(sizeless, mb, len, |end| is_view(va.saturating_add(end - 1)))
}

/// `len`, or `mb` MiB when that is more, the path has no size (`sizeless`) and the last byte
/// of the longer range is still in a view window (`in_view`, by the length).
fn longer(sizeless: bool, mb: u64, len: u64, in_view: impl Fn(u64) -> bool) -> u64 {
    let want = mb << 20;
    if sizeless && want > len && in_view(want) { want } else { len }
}

/// The CPU address behind `va` (itself unless `va` lies in a GPU view of the title's memory,
/// in a view window or not: `sparse::cpu_of` asks `kernelx::mirrors`, a lock). Callers on a hot
/// path ask `is_view` first.
pub fn cpu(va: u64) -> u64 {
    super::sparse::cpu_of(va as usize) as u64
}

/// The CPU address behind `[va, va + len)` (`cpu`) when the range is one piece there: every later
/// 64 KiB page of it lies right behind the first (`cpu(b) == cpu(va) + (b - va)` at each page
/// boundary `b`). None when its pages are kept apart (the queue's page mappings,
/// `kernelx::mirrors`) or one of them lies outside the GPU view.
pub fn cpu_run(va: u64, len: u64) -> Option<u64> {
    let end = va.checked_add(len)?;
    let c = cpu(va);
    let mut b = (va | ((1 << TILE_SHIFT) - 1)).checked_add(1)?;
    while b < end {
        if cpu(b) != c.wrapping_add(b - va) {
            return None;
        }
        b = b.checked_add(1 << TILE_SHIFT)?;
    }
    Some(c)
}

/// (calls, bytes asked for) of each path so far: the 64 KiB tiles each path asked for first
/// (`sparse::map_mirror` queued them), not the length of every range given.
pub fn totals() -> [(u64, u64); PATHS] {
    std::array::from_fn(|i| (CALLS[i].load(Ordering::Relaxed), BYTES[i].load(Ordering::Relaxed)))
}

/// What was asked for between two `totals`: the paths with calls, for the periodic log line.
pub fn line(now: &[(u64, u64); PATHS], before: &[(u64, u64); PATHS]) -> Option<String> {
    let parts: Vec<String> = (0..PATHS)
        .filter(|&i| now[i].0 > before[i].0)
        .map(|i| format!("{} {} ({} KB new)", now[i].0 - before[i].0, NAMES[i], (now[i].1 - before[i].1) >> 10))
        .collect();
    (!parts.is_empty()).then(|| parts.join(", "))
}

#[cfg(test)]
mod reach_tests {
    /// Only a view without a size reaches further, only when asked to, and never past the
    /// view window it is in.
    #[test]
    fn a_sizeless_view_reaches_further_inside_its_window_only() {
        let all = |_: u64| true;
        assert_eq!(super::longer(true, 0, 0x1_0000, all), 0x1_0000, "off");
        assert_eq!(super::longer(true, 64, 0x1_0000, all), 64 << 20);
        assert_eq!(super::longer(false, 64, 0x1_0000, all), 0x1_0000, "a view with a size keeps it");
        assert_eq!(super::longer(true, 1, 8 << 20, all), 8 << 20, "never shorter");
        assert_eq!(super::longer(true, 64, 0x1_0000, |end| end <= 32 << 20), 0x1_0000, "past the window");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The line names only the paths used since the totals before, with their new bytes.
    #[test]
    fn the_line_names_the_paths_used_since() {
        let mut before = [(0u64, 0u64); PATHS];
        let mut now = before;
        assert_eq!(line(&now, &before), None, "nothing asked for: no line");
        now[Path::ViewAt as usize] = (3, 3 << 20);
        now[Path::RootView as usize] = (1, 65536);
        assert_eq!(line(&now, &before).as_deref(), Some("3 views by address (3072 KB new), 1 root views (64 KB new)"));
        before = now;
        now[Path::RootView as usize] = (2, 65536);
        assert_eq!(line(&now, &before).as_deref(), Some("1 root views (0 KB new)"), "a range asked for again adds a call, no bytes");
        assert_eq!(NAMES[Path::Bvh as usize], "acceleration-structure views");
    }

    /// An address outside a view window (every address without the 16-window layout, and null)
    /// asks for nothing and is not counted.
    #[test]
    fn an_address_outside_the_views_asks_for_nothing() {
        let before = totals();
        assert!(!want(Path::Cbv, 0, 256));
        assert!(!want(Path::Cbv, 0x1000, 0));
        assert_eq!(totals(), before);
    }

    /// A thread that asks for the same range again (a vertex buffer bound for every draw), or
    /// for another range in tiles it asked for (a root view at the next offset of a constant
    /// allocator), does not walk its tiles again; a range reaching a tile it did not ask for, a
    /// span pushed out by `LATELY` later ones, and GPU views that changed since make it ask again.
    #[test]
    fn a_range_asked_for_lately_is_not_asked_for_again() {
        std::thread::spawn(|| {
            const A: u64 = 0x14_0000_1000;
            assert!(!asked_lately(A, 0x2000, (7, 0)), "the first time");
            assert!(asked_lately(A, 0x2000, (7, 0)), "again: remembered");
            assert!(asked_lately(0x14_0000_8000, 0x4000, (7, 0)), "another range in the same tile");
            assert!(asked_lately(0x14_0000_0000, 0x1_0000, (7, 0)), "the whole tile");
            assert!(!asked_lately(0x14_0000_8000, 0x1_0000, (7, 0)), "a range reaching the next tile");
            assert!(asked_lately(0x14_0001_0000, 0x100, (7, 0)), "the next tile was asked for with it");
            assert!(asked_lately(0x14_0000_0000, 0x2_0000, (7, 0)), "both tiles");
            assert!(!asked_lately(0x14_0000_0000, 0x2_0001, (7, 0)), "and one byte of a third");
            assert!(!asked_lately(A, 0x2000, (8, 0)), "the views changed: asked again");
            for k in 0..LATELY as u64 - 1 {
                assert!(!asked_lately(0x15_0000_0000 + (k << 16), 0x1000, (8, 0)));
            }
            assert!(asked_lately(A, 0x2000, (8, 0)), "{LATELY} spans are remembered");
            assert!(!asked_lately(0x16_0000_0000, 0x1000, (8, 0)));
            assert!(!asked_lately(A, 0x2000, (8, 0)), "pushed out by {LATELY} later spans");
            assert!(asked_lately(A, 0x2000, (8, 0)));
            // a tile given back since (its source decommitted) may be asked for again: every
            // remembered ask goes, also the ones whose tiles stayed
            assert!(!asked_lately(A, 0x2000, (8, 1)), "an ask was forgotten: asked again");
            assert!(asked_lately(A, 0x2000, (8, 1)));
            assert!(!asked_lately(A, 0x2000, (8, 2)));
        })
        .join()
        .unwrap();
    }

    /// A range of a GPU view is its CPU view's range while the view's pages show the CPU view at
    /// the same offset; a range running past the view's end is not one piece.
    #[test]
    fn a_range_in_a_gpu_view_is_one_piece_only_while_its_pages_are() {
        use crate::hle::kernelx::mirrors;
        const SIZE: usize = 157 << 16;
        // (addresses far above anything Windows hands out: only the pairing is used)
        let (c, g) = (0x72C0_0000_0000u64, 0x72C8_0000_0000u64);
        mirrors::reserved(c as usize, SIZE, 0x2040_2000, 0x04);
        assert!(mirrors::reserved(g as usize, SIZE, 0x2040_2000, 0x801));
        assert_eq!(cpu_run(g + 0x1_2340, 0x100), Some(c + 0x1_2340), "inside one page");
        assert_eq!(cpu_run(g + 0x1_fff0, 0x5_0000), Some(c + 0x1_fff0), "across five page boundaries");
        assert_eq!(cpu_run(g, SIZE as u64), Some(c), "the whole view");
        assert_eq!(cpu_run(g + 0x40, 0), Some(c + 0x40));
        assert_eq!(cpu_run(g + SIZE as u64 - 0x100, 0x100), Some(c + SIZE as u64 - 0x100));
        assert_eq!(cpu_run(g + SIZE as u64 - 0x100, 0x101), None, "past the view's end");
        assert_eq!(cpu_run(g - 0x10, 0x20), None, "from before the view into it");
        assert_eq!(cpu_run(u64::MAX - 0x10, 0x20), None, "past the end of the address space");
        mirrors::released(c as usize);
        assert_eq!(cpu_run(g + 0x1_fff0, 0x20), Some(g + 0x1_fff0), "unpaired: the address itself");
    }
}
