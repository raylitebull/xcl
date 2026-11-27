//! What a ray traversal of the title's acceleration structures reads, asked of the GPU views.
//!
//! The title's shaders walk a top level from its root, jump through each instance node into a
//! bottom level by address and read parent pointers on the way back (`bvh`); only the view of
//! the top level is ever made. So the structures are walked on the CPU side of the views
//! (`bvh::walk_scene`) and every tile reached is asked for (`view_want::want`): when the view is
//! made, and again – once a frame at most – after the CPU wrote a tile a walk read.

use super::view_want::{self, Path};
use crate::gpu::bvh;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

/// top levels kept (the one walked least lately goes)
const TOPS: usize = 64;
/// bottom levels kept: past it all are forgotten (walked again with the next top level)
const BOTTOMS: usize = 1 << 16;
/// top levels walked again a frame
const A_FRAME: usize = 2;
const TILE: u64 = bvh::WALK_TILE;

struct Top {
    va: u64,
    /// tiles of its own walk, and the CPU tiles they show
    tiles: Vec<(u64, u64)>,
    cpu: Vec<u64>,
    referenced: Vec<u64>,
    stale: bool,
    walking: bool,
    /// the frame of its last walk, and `view_want::generation` then
    frame: u64,
    gen: (u64, u64),
}

struct Bottom {
    tiles: Vec<(u64, u64)>,
    cpu: Vec<u64>,
    nodes: u64,
    stale: bool,
}

#[derive(Default)]
struct Reg {
    tops: Vec<Top>,
    bottoms: HashMap<u64, Bottom>,
    /// CPU tile (index) → the structures walked there (top level: address, bottom level: | 1)
    watch: HashMap<u64, Vec<u64>>,
    frame: u64,
    bottoms_stale: bool,
    /// writes found where a walk read (`mark_written`), so far
    writes: u64,
}

static REG: LazyLock<Mutex<Reg>> = LazyLock::new(Default::default);
/// whether `Reg::watch` has a tile
static WATCHING: AtomicBool = AtomicBool::new(false);

fn with<R>(f: impl FnOnce(&mut Reg) -> R) -> R {
    f(&mut REG.lock().unwrap_or_else(|e| e.into_inner()))
}

/// `XCL_BVH_WALK=0`: no walk, only the 64 KiB at a structure's address (as before)
fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        let on = std::env::var("XCL_BVH_WALK").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: XCL_BVH_WALK=0: acceleration structures are asked for 64 KiB from their address only");
        }
        on
    });
    *ON && super::bind::view_span().1 != 0
}

/// The title's bytes behind GPU address `va` (its CPU view), false when any cannot be read (a
/// page not committed, or decommitted while read: the kernel's copy fails, nothing faults).
fn read_title(va: u64, out: &mut [u8]) -> bool {
    let Some(c) = view_want::cpu_run(va, out.len() as u64) else { return false };
    unsafe { crate::hle::dstorage::read_own(c as usize, out.as_mut_ptr(), out.len()) }
}

fn ask(tiles: &[(u64, u64)]) {
    for &(at, len) in tiles {
        view_want::want(Path::Bvh, at, len);
    }
}

/// The CPU tiles (indices) the GPU tiles a walk of the structure at `base` reached show (its
/// header's when it reached none: a write there makes it worth another walk)
fn cpu_tiles(base: u64, tiles: &[(u64, u64)]) -> Vec<u64> {
    let header = [(base & !(TILE - 1), TILE)];
    let tiles = if tiles.is_empty() { &header[..] } else { tiles };
    let mut out: Vec<u64> = tiles
        .iter()
        .flat_map(|&(at, len)| (0..len / TILE).map(move |k| at + k * TILE))
        .flat_map(|t| {
            let c = view_want::cpu(t);
            [c / TILE, (c + TILE - 1) / TILE]
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn watch(w: &mut HashMap<u64, Vec<u64>>, cpu: &[u64], key: u64, on: bool) {
    for t in cpu {
        let keys = w.entry(*t).or_default();
        keys.retain(|&k| k != key);
        if on {
            keys.push(key);
        } else if keys.is_empty() {
            w.remove(t);
        }
    }
}

/// An acceleration-structure view of the top level at `va` was made: what rays can read of it
/// and of its bottom levels is asked for; walked again only when the CPU wrote there since.
pub fn view_made(va: u64) {
    if !on() {
        return;
    }
    let gen = view_want::generation();
    let known = with(|r| {
        let frame = r.frame;
        let t = r.tops.iter_mut().find(|t| t.va == va)?;
        if t.walking || (t.stale && t.frame == frame) {
            return Some(Vec::new());
        }
        if t.stale {
            return None;
        }
        // asked for again only when asks were forgotten since
        if t.gen == gen {
            return Some(Vec::new());
        }
        t.gen = gen;
        let mut tiles = t.tiles.clone();
        let Reg { tops, bottoms, .. } = r;
        let t = tops.iter().find(|t| t.va == va)?;
        tiles.extend(t.referenced.iter().filter_map(|a| bottoms.get(a)).flat_map(|b| b.tiles.iter().copied()));
        Some(tiles)
    });
    match known {
        Some(tiles) => ask(&tiles),
        None => walk(va),
    }
}

/// Once a frame: the top levels the CPU wrote (or whose bottom levels it wrote) since their
/// walk are walked again, `A_FRAME` at most, the least lately walked first.
pub fn frame() {
    if !on() {
        return;
    }
    for va in with(due) {
        walk(va);
    }
}

/// `frame` with the registry locked: the top levels to walk again
fn due(r: &mut Reg) -> Vec<u64> {
    r.frame += 1;
    if std::mem::take(&mut r.bottoms_stale) {
        let Reg { tops, bottoms, .. } = r;
        for t in tops.iter_mut() {
            t.stale |= t.referenced.iter().any(|a| bottoms.get(a).is_some_and(|b| b.stale));
        }
    }
    let mut due: Vec<(u64, u64)> = r.tops.iter().filter(|t| t.stale && !t.walking).map(|t| (t.frame, t.va)).collect();
    due.sort_unstable();
    due.into_iter().take(A_FRAME).map(|d| d.1).collect()
}

/// Before a submit, the CPU pages written (`page` bytes each): the structures walked there are
/// walked again (`frame`).
pub fn written(dirty: &[usize], page: usize) {
    if !WATCHING.load(Ordering::Relaxed) || dirty.is_empty() {
        return;
    }
    with(|r| mark_written(r, dirty, page as u64));
}

fn mark_written(r: &mut Reg, dirty: &[usize], page: u64) {
    let mut keys = Vec::new();
    let mut last = u64::MAX;
    for &a in dirty {
        for t in a as u64 / TILE..=(a as u64 + page.max(1) - 1) / TILE {
            if t != last {
                last = t;
                keys.extend(r.watch.get(&t).into_iter().flatten().copied());
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    r.writes += !keys.is_empty() as u64;
    for k in keys {
        if k & 1 != 0 {
            if let Some(b) = r.bottoms.get_mut(&(k & !1)) {
                b.stale = true;
                r.bottoms_stale = true;
            }
        } else if let Some(t) = r.tops.iter_mut().find(|t| t.va == k) {
            t.stale = true;
        }
    }
}

fn walk(va: u64) {
    let frame = with(|r| {
        if r.tops.iter().any(|t| t.va == va && t.walking) {
            return None;
        }
        if !r.tops.iter().any(|t| t.va == va) {
            if r.tops.len() >= TOPS {
                if let Some(i) = (0..r.tops.len()).filter(|&i| !r.tops[i].walking).min_by_key(|&i| r.tops[i].frame) {
                    let t = r.tops.swap_remove(i);
                    watch(&mut r.watch, &t.cpu, t.va, false);
                }
            }
            r.tops.push(Top { va, tiles: Vec::new(), cpu: Vec::new(), referenced: Vec::new(), stale: true, walking: false, frame: 0, gen: (0, 0) });
        }
        r.tops.iter_mut().find(|t| t.va == va)?.walking = true;
        Some((r.frame, r.writes))
    });
    let Some((frame, writes)) = frame else { return };
    let t0 = std::time::Instant::now();
    let s = bvh::walk_scene(&mut read_title, va, &bvh::Limits::default(), &mut |b| with(|r| r.bottoms.get(&b).is_none_or(|x| x.stale)));
    let us = t0.elapsed().as_micros();
    let gen = view_want::generation();
    let mut tiles: Vec<(u64, u64)> = s.top.tiles.iter().chain(s.bottoms.iter().flat_map(|b| &b.tiles)).copied().collect();
    let top_cpu = cpu_tiles(va, &s.top.tiles);
    let bottoms_cpu: Vec<Vec<u64>> = s.bottoms.iter().map(|b| cpu_tiles(b.base, &b.tiles)).collect();
    let (nodes, bottom_bytes) = with(|r| {
        if r.bottoms.len() + s.bottoms.len() > BOTTOMS {
            r.bottoms.clear();
            r.watch.retain(|_, keys| {
                keys.retain(|k| k & 1 == 0);
                !keys.is_empty()
            });
        }
        // (a write the walk may have missed: all of it again next frame)
        let missed = r.writes != writes;
        for (b, cpu) in s.bottoms.iter().zip(bottoms_cpu) {
            if let Some(old) = r.bottoms.remove(&b.base) {
                watch(&mut r.watch, &old.cpu, b.base | 1, false);
            }
            watch(&mut r.watch, &cpu, b.base | 1, true);
            // (one with no node yet is walked again with its top level)
            r.bottoms.insert(b.base, Bottom { tiles: b.tiles.clone(), cpu, nodes: b.nodes, stale: b.nodes == 0 || missed });
        }
        let Reg { tops, bottoms, watch: w, .. } = r;
        let Some(t) = tops.iter_mut().find(|t| t.va == va) else { return (0, 0) };
        watch(w, &t.cpu, va, false);
        watch(w, &top_cpu, va, true);
        let known: Vec<&Bottom> = s.referenced.iter().filter_map(|a| bottoms.get(a)).collect();
        // (the bottom levels known already: their tiles again once asks were forgotten since)
        if t.gen != gen {
            let walked: std::collections::HashSet<u64> = s.bottoms.iter().map(|b| b.base).collect();
            tiles.extend(s.referenced.iter().filter(|a| !walked.contains(a)).filter_map(|a| bottoms.get(a)).flat_map(|b| b.tiles.iter().copied()));
        }
        // a top level with no node yet (not written, or written by the GPU only) is walked again
        *t = Top { va, tiles: s.top.tiles.clone(), cpu: top_cpu, referenced: s.referenced.clone(), stale: s.top.nodes == 0 || missed, walking: false, frame, gen };
        WATCHING.store(!w.is_empty(), Ordering::Relaxed);
        (s.top.nodes + known.iter().map(|b| b.nodes).sum::<u64>(), known.iter().flat_map(|b| &b.tiles).map(|x| x.1).sum::<u64>())
    });
    ask(&tiles);
    static WALKS: AtomicU64 = AtomicU64::new(0);
    let n = WALKS.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let top_kib = s.top.tiles.iter().map(|x| x.1).sum::<u64>() >> 10;
        let cut = s.cut.map_or(String::new(), |c| format!(", truncated: {c}"));
        let parents = if s.top.nodes != 0 && s.top.parents.is_none() { ", no parent array" } else { "" };
        crate::xlog_info!(
            "D3D12.X: acceleration structure at {va:#x}: {nodes} nodes, {top_kib} KiB of top level, {} bottom levels ({} KiB) asked for; walk {us} µs{cut}{parents} (walk {n})",
            s.referenced.len(),
            bottom_bytes >> 10
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn top(va: u64, cpu: Vec<u64>, referenced: Vec<u64>, frame: u64) -> Top {
        Top { va, tiles: Vec::new(), cpu, referenced, stale: false, walking: false, frame, gen: (0, 0) }
    }

    /// A CPU write in a tile a walk read makes that structure stale – a bottom level's makes the
    /// top levels reaching it walked again – and a write elsewhere does nothing; a frame walks
    /// `A_FRAME` of them at most, the least lately walked first.
    #[test]
    fn a_write_where_a_walk_read_walks_it_again_once_a_frame() {
        const T: u64 = 0x10_0000;
        const B: u64 = 0x20_0000;
        let mut r = Reg::default();
        for (k, f) in [(0u64, 5u64), (1, 3), (2, 4)] {
            r.tops.push(top(T + k * 0x100, vec![T / TILE + k], vec![B], f));
            watch(&mut r.watch, &[T / TILE + k], T + k * 0x100, true);
        }
        r.bottoms.insert(B, Bottom { tiles: Vec::new(), cpu: vec![B / TILE, B / TILE + 1], nodes: 9, stale: false });
        watch(&mut r.watch, &[B / TILE, B / TILE + 1], B | 1, true);
        mark_written(&mut r, &[(T + 3 * TILE) as usize, (B + 5 * TILE) as usize], 0x1000);
        assert!(r.tops.iter().all(|t| !t.stale) && !r.bottoms[&B].stale, "written elsewhere");
        assert!(due(&mut r).is_empty());
        mark_written(&mut r, &[(T + TILE + 0x3000) as usize], 0x1000);
        assert!(r.tops[1].stale && !r.tops[0].stale && !r.tops[2].stale);
        assert_eq!(due(&mut r), [T + 0x100]);
        r.tops[1].stale = false;
        // a page reaching into the bottom level's second tile
        mark_written(&mut r, &[(B + TILE - 0x1000) as usize], 0x2000);
        assert!(r.bottoms[&B].stale);
        assert_eq!(due(&mut r), [T + 0x100, T + 0x200], "all three reach it: two a frame, least lately walked first");
        // a structure walked again where it is no more
        watch(&mut r.watch, &[T / TILE], T, false);
        r.tops[0].stale = false;
        mark_written(&mut r, &[T as usize], 0x1000);
        assert!(!r.tops[0].stale);
        assert!(!r.watch.contains_key(&(T / TILE)));
    }
}
