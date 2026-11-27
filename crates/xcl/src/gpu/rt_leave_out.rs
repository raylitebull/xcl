//! `XCL_RT_LEAVE_OUT=<fragment>[,<fragment>…]` (off by default): compute programs whose shader
//! container holds one of the fragments are left out – the title's pipeline gets the dummy
//! compute program a console shader that cannot be translated gets, so its dispatches are
//! skipped, and the shader preparation and `xcl shaders --precompile` pass them over.
//!
//! What it is for: a title's ray-tracing programs that shade the hits of rays or start rays
//! (on a retail title about 1,000 programs, ~1 MB of DXIL each, the largest of its cold driver
//! compiles) do nothing while the title has built no acceleration structure: every ray misses.
//! The fragments name those programs (a name in the container's debug data; find it with
//! `xcl rpf <pack> --find`); XCL's source names no title's programs.
//!
//! So that leaving them out stays harmless, XCL watches the acceleration structures the
//! title's views name (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE` in unified
//! memory) and warns once when one of them gets a root node: from then on rays can hit, and
//! the programs left out would have shaded those hits.
//!
//! Ported from Vice-Tay (https://github.com/TaynanGT/Vice-Tay, GPL-3.0-or-later), commit
//! `1f69d67`, `crates/victory-gpu/src/pipeline/make.rs` (`left_out`: `VICTORY_RT_RESOLVE`,
//! `VICTORY_ACTIVATE_RAYS`, `VICTORY_BVH_BUILDER`; there since codec_d's `8cc31ab`).
//! Changes (2026-11-27): the programs are named by the user's fragments instead of names in
//! the source; one switch; the structure watch and its warning are new; the preparation and
//! the precompile pass the programs over.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Fragments shorter than this are refused: they would match bytes of any container.
const SHORTEST: usize = 6;
/// Acceleration structures remembered for the watch (distinct addresses).
const MOST_STRUCTURES: usize = 4096;

/// The fragments of a `XCL_RT_LEAVE_OUT` value: comma-separated, trimmed, at least
/// [`SHORTEST`] bytes each (shorter ones are dropped).
pub fn fragments_of(value: &str) -> Vec<Vec<u8>> {
    value.split(',').map(str::trim).filter(|f| f.len() >= SHORTEST).map(|f| f.as_bytes().to_vec()).collect()
}

/// Whether `container` holds one of `fragments`.
pub fn holds_any(container: &[u8], fragments: &[Vec<u8>]) -> bool {
    fragments.iter().any(|f| container.windows(f.len()).any(|w| w == f.as_slice()))
}

fn fragments() -> &'static [Vec<u8>] {
    static F: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    F.get_or_init(|| {
        let Ok(v) = std::env::var("XCL_RT_LEAVE_OUT") else { return Vec::new() };
        let f = fragments_of(&v);
        let dropped = v.split(',').map(str::trim).filter(|s| !s.is_empty() && s.len() < SHORTEST).count();
        if f.is_empty() {
            crate::xlog_warn!("XCL_RT_LEAVE_OUT: no fragment of at least {SHORTEST} bytes – nothing is left out");
        } else {
            crate::xlog_info!(
                "XCL_RT_LEAVE_OUT: compute programs holding one of {} fragment(s) are left out while rays cannot hit{}",
                f.len(),
                if dropped > 0 { format!(" ({dropped} shorter than {SHORTEST} bytes ignored)") } else { String::new() }
            );
        }
        f
    })
}

/// Whether the switch is on (at least one usable fragment).
pub fn enabled() -> bool {
    !fragments().is_empty()
}

static LEFT_OUT: AtomicU64 = AtomicU64::new(0);

/// Whether the compute program in `container` is left out; counts and logs the first few.
pub fn left_out(container: &[u8]) -> bool {
    if !enabled() || !holds_any(container, fragments()) {
        return false;
    }
    let n = LEFT_OUT.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 3 || n.is_power_of_two() {
        crate::xlog_info!(
            "XCL_RT_LEAVE_OUT: compute program of record {:016x} left out (#{n}): its dispatches are skipped",
            crate::gpu::shader::fnv64(container)
        );
    }
    true
}

/// How many compute programs were left out so far.
pub fn left_out_count() -> u64 {
    LEFT_OUT.load(Ordering::Relaxed)
}

static STRUCTURES: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());

/// An acceleration structure in unified memory that a title's view names (only kept while the
/// switch is on).
pub fn note_structure(va: u64) {
    if enabled() {
        let mut s = STRUCTURES.lock().unwrap();
        if s.len() < MOST_STRUCTURES {
            s.insert(va);
        }
    }
}

/// The root node id in an acceleration structure's header: `None` while it has none (0 or
/// all ones, as before a build).
pub fn root_of(header: &[u8]) -> Option<u32> {
    let at = crate::gpu::bvh::ROOT_AT;
    let root = u32::from_le_bytes(header.get(at..at + 4)?.try_into().ok()?);
    (root != 0 && root != u32::MAX).then_some(root)
}

/// Looks at the structures views named (called every few hundred frames): warns once when one
/// has a root node, since the programs left out would then have hits to shade.
pub fn check_structures() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !enabled() || WARNED.load(Ordering::Relaxed) {
        return;
    }
    let all: Vec<u64> = STRUCTURES.lock().unwrap().iter().copied().collect();
    let header = crate::gpu::bvh::HEADER_BYTES;
    for va in all {
        let p = va as usize;
        if !crate::gpu::shader::readable(p, header) {
            continue;
        }
        // SAFETY: `readable` just checked the range; title memory, only read
        let bytes = unsafe { std::slice::from_raw_parts(p as *const u8, header) };
        if let Some(root) = root_of(bytes) {
            if !WARNED.swap(true, Ordering::Relaxed) {
                crate::xlog_warn!(
                    "XCL_RT_LEAVE_OUT: the acceleration structure at {va:#x} has a root node ({root:#x}) now – rays can hit, and the {} compute program(s) left out would shade those hits; run without XCL_RT_LEAVE_OUT for this title",
                    left_out_count()
                );
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragments_are_trimmed_and_short_ones_dropped() {
        let f = fragments_of(" ResolveA ,abc, ,LongerName,");
        assert_eq!(f, vec![b"ResolveA".to_vec(), b"LongerName".to_vec()]);
        assert!(fragments_of("").is_empty());
        assert!(fragments_of("short").is_empty());
    }

    #[test]
    fn a_container_is_matched_by_any_fragment_anywhere() {
        let mut container = vec![0u8; 64];
        container.extend_from_slice(b"\x00prog_Example_Name_x\x00");
        container.extend_from_slice(&[0xFF; 32]);
        assert!(holds_any(&container, &fragments_of("Other_Name,Example_Name")));
        assert!(!holds_any(&container, &fragments_of("Example_Names")));
        assert!(!holds_any(&container, &[]));
        // a fragment at the very end still matches
        assert!(holds_any(b"xxxxEnd_of_it", &fragments_of("End_of_it")));
    }

    #[test]
    fn a_header_without_a_root_has_none() {
        let mut h = vec![0u8; crate::gpu::bvh::HEADER_BYTES];
        assert_eq!(root_of(&h), None);
        let at = crate::gpu::bvh::ROOT_AT;
        h[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(root_of(&h), None);
        h[at..at + 4].copy_from_slice(&0x25u32.to_le_bytes());
        assert_eq!(root_of(&h), Some(0x25));
        assert_eq!(root_of(&h[..at]), None, "a header too short to hold the root");
    }

    #[test]
    fn a_built_structure_from_the_builder_has_a_root() {
        // XCL's own builder writes the console layout the watch reads
        let tris = crate::gpu::bvh::tests::random_tris(8, 3);
        let b = crate::gpu::bvh::build(&tris, 0);
        let root = root_of(&b.data).expect("a built structure has a root id");
        assert_eq!(root, b.root);
    }
}
