//! Command list #146 `Barrier(groups)`: a title's enhanced barriers as the PC's transitions.
//!
//! A title that names layouts instead of states makes all its transitions through this call (a
//! retail title's frame held 15 `ResourceBarrier` calls, its upscaler's, and thousands of these).
//! Answered with a UAV barrier alone, the PC's textures stayed in the state they were made in:
//! a render target was sampled without the driver ever taking it out of its compressed form, a
//! depth buffer read as a texture was never expanded, and a depth buffer the title moved into
//! unordered access never reached `depth_uav`, whose copies between a depth plane and its twin
//! hang on that transition.
//!
//! Each texture barrier becomes a legacy transition between the states its layouts stand for
//! (`state_of`), on the subresources it names, and goes the way a title's `ResourceBarrier`
//! goes (`list_x::resource_barrier_x`). A layout says where a texture is; the access bits only
//! say which uses the title makes of it there, so the states come from the layouts. A barrier to
//! the undefined layout (the title is done with what the texture holds) moves nothing; one from
//! it starts at the state the texture was last moved to here. Buffer and global barriers have no
//! layout: the UAV barrier `list_x::uav_barrier_x` sends stands for them.
//!
//! `XCL_BARRIER_GROUPS=0` turns the transitions off (the UAV barrier stays).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::sharded::Sharded;

/// D3D12_BARRIER_LAYOUT_UNDEFINED
const UNDEFINED: u32 = 0xFFFF_FFFF;
/// D3D12_BARRIER_SYNC_SPLIT
const SYNC_SPLIT: u32 = 0x8000_0000;
/// D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ
const ACCESS_DEPTH_READ: u32 = 0x40;
/// D3D12_RESOURCE_STATE_*
const RENDER_TARGET: u32 = 0x4;
const UNORDERED_ACCESS: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;
const RESOLVE_DEST: u32 = 0x1000;
const RESOLVE_SOURCE: u32 = 0x2000;
const SHADING_RATE_SOURCE: u32 = 0x100_0000;
/// D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, END_ONLY
const BEGIN_ONLY: u32 = 1;
const END_ONLY: u32 = 2;
/// subresources of one barrier's range named one by one at most (more: all of the texture)
const RANGE_MOST: usize = 64;

/// Whether the title's barrier groups become transitions (`XCL_BARRIER_GROUPS`, on unless "0").
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_BARRIER_GROUPS").map_or(true, |v| v.trim() != "0");
        if !on {
            crate::xlog_info!("D3D12.X: XCL_BARRIER_GROUPS=0 – the title's barrier groups (#146) move no texture");
        }
        on
    })
}

/// The state the PC has a texture in where the title has it in `layout` with `access`
/// (`depth`: a depth buffer). None: the undefined layout, or one XCL does not know.
pub(super) fn state_of(layout: u32, access: u32, depth: bool) -> Option<u32> {
    // reading layouts take a depth buffer's read-only depth use with them
    let read = |s: u32| if depth && (access & ACCESS_DEPTH_READ != 0 || access & 0x8000_0000 != 0) { s | DEPTH_READ } else { s };
    Some(match layout {
        // COMMON / PRESENT, and the queues' own
        0 | 18 | 24 | 30 => 0,
        // GENERIC_READ, DIRECT_QUEUE_GENERIC_READ, COMPUTE_QUEUE_GENERIC_READ,
        // DIRECT_QUEUE_GENERIC_READ_COMPUTE_QUEUE_ACCESSIBLE
        1 | 19 | 25 | 31 => read(SHADER_RESOURCE | COPY_SOURCE),
        2 => RENDER_TARGET,
        3 | 20 | 26 => UNORDERED_ACCESS,
        4 => DEPTH_WRITE,
        // (read by the depth test and by shaders at once)
        5 => DEPTH_READ | SHADER_RESOURCE,
        6 | 21 | 27 => read(SHADER_RESOURCE),
        7 | 22 | 28 => COPY_SOURCE,
        8 | 23 | 29 => COPY_DEST,
        9 => RESOLVE_SOURCE,
        10 => RESOLVE_DEST,
        11 => SHADING_RATE_SOURCE,
        _ => return None,
    })
}

/// The subresources a barrier's range names in a texture of `mips` levels, `array` slices and
/// `planes` planes: `range` is D3D12_BARRIER_SUBRESOURCE_RANGE (index or first level, levels,
/// first slice, slices, first plane, planes; no levels: the first word is a subresource index).
/// None: all of them.
pub(super) fn subresources(range: [u32; 6], mips: u32, array: u32, planes: u32) -> Option<Vec<u32>> {
    let [first_mip, n_mips, first_slice, n_slices, first_plane, n_planes] = range;
    let (mips, array, planes) = (mips.max(1), array.max(1), planes.max(1));
    if n_mips == 0 {
        return (first_mip != u32::MAX && (mips * array * planes > 1)).then(|| vec![first_mip]);
    }
    // (slices and planes of 0 are one, as the PC reads the range)
    let m = first_mip.min(mips)..first_mip.saturating_add(n_mips).min(mips);
    let s = first_slice.min(array)..first_slice.saturating_add(n_slices.max(1)).min(array);
    let p = first_plane.min(planes)..first_plane.saturating_add(n_planes.max(1)).min(planes);
    let count = m.len() * s.len() * p.len();
    if count == (mips * array * planes) as usize || count > RANGE_MOST {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    for plane in p {
        for slice in s.clone() {
            for mip in m.clone() {
                out.push(mip + slice * mips + plane * mips * array);
            }
        }
    }
    Some(out)
}

/// One texture barrier as the words of a D3D12_TEXTURE_BARRIER the PC reads it by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Texture {
    pub sync: (u32, u32),
    pub access: (u32, u32),
    pub layout: (u32, u32),
    pub res: usize,
    pub range: [u32; 6],
}

/// What a texture is like for `transitions_of`: (depth buffer, levels, slices, planes).
pub(super) type Shape = (bool, u32, u32, u32);

/// The transitions (PC D3D12_RESOURCE_BARRIERs as words) of the title's texture barriers `b`:
/// `shape` gives a texture's shape (None: not a texture XCL can ask), `last` the state a
/// texture was last moved to here, and `moved` notes the one it is in after a barrier.
pub(super) fn transitions_of(b: &[Texture], shape: &mut dyn FnMut(usize) -> Option<Shape>, last: &mut dyn FnMut(usize) -> Option<u32>, moved: &mut dyn FnMut(usize, u32)) -> Vec<[u32; 8]> {
    let mut out = Vec::with_capacity(b.len());
    for t in b {
        if t.res == 0 {
            continue;
        }
        let Some((depth, mips, array, planes)) = shape(t.res) else { continue };
        let after = state_of(t.layout.1, t.access.1, depth);
        let before = match t.layout.0 {
            UNDEFINED => last(t.res),
            l => state_of(l, t.access.0, depth),
        };
        if t.layout.1 != UNDEFINED && after.is_none() || t.layout.0 != UNDEFINED && before.is_none() {
            unknown_layout(t.layout);
            continue;
        }
        let (before, after) = match (before, after) {
            (Some(b), Some(a)) => (b, a),
            // to the undefined layout: the texture stays where it is; from it, never moved
            // here: where it goes is where it is taken to be
            (b, a) => {
                if let Some(s) = b.or(a) {
                    moved(t.res, s);
                }
                continue;
            }
        };
        moved(t.res, after);
        if before == after {
            continue;
        }
        // a barrier split over two lists: its first half ends in SYNC_SPLIT, its second starts there
        let flags = if t.sync.1 == SYNC_SPLIT { BEGIN_ONLY } else if t.sync.0 == SYNC_SPLIT { END_ONLY } else { 0 };
        let word = |sub: u32| [0, flags, t.res as u32, (t.res as u64 >> 32) as u32, sub, before, after, 0];
        match subresources(t.range, mips, array, planes) {
            None => out.push(word(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)),
            Some(subs) => out.extend(subs.into_iter().map(word)),
        }
    }
    out
}

fn unknown_layout(layout: (u32, u32)) {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a barrier group's texture barrier between layouts {:#x} and {:#x}, one of which XCL does not know: the texture is not moved ({n} so far)", layout.0, layout.1);
    }
}

/// The state each texture was last moved to by a barrier group (the title's recording order).
static LAST: Sharded<u32> = Sharded::new();
/// Texture barriers read, and transitions made of them.
static COUNTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The last reference of `res` goes (`real::res_release`).
pub(super) fn forget(res: usize) {
    if COUNTS[0].load(Ordering::Relaxed) != 0 {
        LAST.remove(res);
    }
}

/// `state` as a texture with the PC resource flags `flags` (D3D12_RESOURCE_FLAGS) can be in it on
/// a direct list (`direct`) or a compute one: the reading states the texture or the list cannot
/// have are left out of it. None: it cannot be there at all (a render target state of a texture
/// that is none on the PC, a depth state on a compute list) – the PC refuses such a transition
/// when the list closes, and the title stops on that.
pub(super) fn legal(state: u32, flags: u32, direct: bool) -> Option<u32> {
    const ALLOW_RENDER_TARGET: u32 = 0x1;
    const ALLOW_DEPTH_STENCIL: u32 = 0x2;
    const ALLOW_UNORDERED_ACCESS: u32 = 0x4;
    const DENY_SHADER_RESOURCE: u32 = 0x8;
    let mut s = state;
    if s & RENDER_TARGET != 0 && flags & ALLOW_RENDER_TARGET == 0 || s & UNORDERED_ACCESS != 0 && flags & ALLOW_UNORDERED_ACCESS == 0 || s & DEPTH_WRITE != 0 && flags & ALLOW_DEPTH_STENCIL == 0 {
        return None;
    }
    if flags & ALLOW_DEPTH_STENCIL == 0 {
        s &= !DEPTH_READ;
    }
    if flags & DENY_SHADER_RESOURCE != 0 {
        s &= !SHADER_RESOURCE;
    }
    if !direct {
        if s & (RENDER_TARGET | DEPTH_WRITE | DEPTH_READ | RESOLVE_DEST | RESOLVE_SOURCE | SHADING_RATE_SOURCE) != 0 {
            return None;
        }
        // (PIXEL_SHADER_RESOURCE)
        s &= !0x80;
    }
    Some(s)
}

/// What `transitions` asks of a texture once: its flags, its subresources, and its format.
#[derive(Clone, Copy)]
struct Asked {
    flags: u32,
    subresources: u32,
    format: i32,
}

/// The title's `n` barrier groups at `groups` (D3D12_BARRIER_GROUP) as transitions for the PC
/// list (`direct`: a direct one). Nothing with the switch off.
pub(super) unsafe fn transitions(n: u32, groups: *const c_void, direct: bool) -> Vec<[u32; 8]> {
    if !enabled() || n == 0 || groups.is_null() {
        return Vec::new();
    }
    let count = (n as usize).min(256);
    if !crate::gpu::shader::readable(groups as usize, count * std::mem::size_of::<D3D12_BARRIER_GROUP>()) {
        return Vec::new();
    }
    let mut textures = Vec::new();
    for grp in std::slice::from_raw_parts(groups as *const D3D12_BARRIER_GROUP, count) {
        let nb = (grp.NumBarriers as usize).min(4096);
        let p = grp.Anonymous.pTextureBarriers;
        if grp.Type != D3D12_BARRIER_TYPE_TEXTURE || nb == 0 || p.is_null() || !crate::gpu::shader::readable(p as usize, nb * std::mem::size_of::<D3D12_TEXTURE_BARRIER>()) {
            continue;
        }
        for tb in std::slice::from_raw_parts(p, nb) {
            let r = &tb.Subresources;
            textures.push(Texture {
                sync: (tb.SyncBefore.0 as u32, tb.SyncAfter.0 as u32),
                access: (tb.AccessBefore.0 as u32, tb.AccessAfter.0 as u32),
                layout: (tb.LayoutBefore.0 as u32, tb.LayoutAfter.0 as u32),
                res: std::mem::transmute_copy(&tb.pResource),
                range: [r.IndexOrFirstMipLevel, r.NumMipLevels, r.FirstArraySlice, r.NumArraySlices, r.FirstPlane, r.NumPlanes],
            });
        }
    }
    if textures.is_empty() {
        return Vec::new();
    }
    let mut asked: Vec<(usize, Asked)> = Vec::new();
    let mut out = transitions_of(
        &textures,
        &mut |res| {
            if !crate::gpu::shader::readable(res, 8) {
                return None;
            }
            let r = res as *mut c_void;
            let d = ID3D12Resource::from_raw_borrowed(&r)?.GetDesc();
            if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                return None;
            }
            let depth = d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0;
            let array = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { d.DepthOrArraySize as u32 };
            // (two planes: the depth-stencil formats R32G8X24 and R24G8 and their views)
            let planes = if matches!(d.Format.0, 19..=22 | 44..=47) { 2 } else { 1 };
            if !asked.iter().any(|a| a.0 == res) {
                asked.push((res, Asked { flags: d.Flags.0 as u32, subresources: d.MipLevels as u32 * array * planes, format: d.Format.0 }));
            }
            Some((depth, d.MipLevels as u32, array, planes))
        },
        &mut |res| LAST.get(res),
        &mut |res, state| {
            LAST.insert(res, state);
        },
    );
    out.retain_mut(|w| {
        let res = (w[2] as u64 | (w[3] as u64) << 32) as usize;
        let Some(a) = asked.iter().find(|a| a.0 == res).map(|a| a.1) else { return false };
        if w[4] != D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES && w[4] >= a.subresources {
            return refused(w, a, "a subresource the texture does not have");
        }
        // a depth buffer the PC gave no unordered access (`device_x::pc_resource_desc`) cannot
        // be in UNORDERED_ACCESS: one `depth_uav` keeps is taken from there by
        // `depth_uav::plan_barriers`, any other rests where that would park it
        let depth_only = a.flags & 0x2 != 0 && a.flags & 0x4 == 0;
        let kept = depth_only && super::depth_uav::kept(res);
        for i in 5..7 {
            if depth_only && w[i] == UNORDERED_ACCESS {
                if !kept {
                    w[i] = super::depth_uav::PARK;
                }
                continue;
            }
            match legal(w[i], a.flags, direct) {
                Some(s) => w[i] = s,
                None => return refused(w, a, if direct { "a state the PC's texture cannot be in" } else { "a state a compute list cannot move a texture to or from" }),
            }
        }
        w[5] != w[6]
    });
    let read = COUNTS[0].fetch_add(textures.len() as u64, Ordering::Relaxed) + textures.len() as u64;
    let made = COUNTS[1].fetch_add(out.len() as u64, Ordering::Relaxed) + out.len() as u64;
    if (read - textures.len() as u64).leading_zeros() != read.leading_zeros() && read >= 1024 {
        crate::xlog_info!("D3D12.X: barrier groups (#146): {read} texture barriers of the title so far, {made} transitions made of them, {} not made (the PC would refuse them)", REFUSED.load(Ordering::Relaxed));
    }
    if log_each() {
        for w in &out {
            let res = (w[2] as u64 | (w[3] as u64) << 32) as usize;
            if let Some(a) = asked.iter().find(|a| a.0 == res).map(|a| a.1) {
                static SEEN: std::sync::Mutex<Option<std::collections::HashSet<(u32, i32, u32, u32, u32, u32, bool)>>> = std::sync::Mutex::new(None);
                let mut g = SEEN.lock().unwrap();
                let seen = g.get_or_insert_with(Default::default);
                if seen.len() < 400 && seen.insert((a.flags, a.format, w[4], w[5], w[6], w[1], direct)) {
                    crate::xlog_info!("D3D12.X: barrier groups: a transition of a kind not seen before: texture {res:#x} (flags {:#x}, format {}, {} subresources), subresource {:#x}, {:#x} → {:#x}, flags {:#x}, on a {} list", a.flags, a.format, a.subresources, w[4], w[5], w[6], w[1], if direct { "direct" } else { "compute or copy" });
                }
            }
        }
    }
    out
}

/// Transitions not made: the PC would refuse them.
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// A transition that is not made (false, for `retain_mut`).
fn refused(w: &[u32; 8], a: Asked, why: &str) -> bool {
    let n = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: barrier groups: a transition of a texture (flags {:#x}, format {}, {} subresources) not made – {why}: subresource {:#x}, {:#x} → {:#x} ({n} so far)", a.flags, a.format, a.subresources, w[4], w[5], w[6]);
    }
    false
}

/// `XCL_BARRIER_LOG=1`: each kind of transition made, once.
fn log_each() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_BARRIER_LOG").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    const NO_ACCESS: u32 = 0x8000_0000;

    #[test]
    fn a_layout_stands_for_the_state_the_pc_has_the_texture_in() {
        assert_eq!(state_of(2, 0x8, false), Some(RENDER_TARGET));
        assert_eq!(state_of(6, 0x80, false), Some(SHADER_RESOURCE));
        assert_eq!(state_of(21, 0x80, false), Some(SHADER_RESOURCE), "the direct queue's own");
        assert_eq!(state_of(3, 0x10, true), Some(UNORDERED_ACCESS));
        assert_eq!(state_of(4, 0x20, true), Some(DEPTH_WRITE));
        assert_eq!(state_of(5, 0x40, true), Some(DEPTH_READ | SHADER_RESOURCE));
        assert_eq!(state_of(6, 0x80, true), Some(SHADER_RESOURCE), "a depth buffer only shaders read");
        assert_eq!(state_of(6, 0x80 | 0x40, true), Some(SHADER_RESOURCE | DEPTH_READ));
        assert_eq!(state_of(31, NO_ACCESS, false), Some(SHADER_RESOURCE | COPY_SOURCE));
        assert_eq!(state_of(0, 0, false), Some(0));
        assert_eq!(state_of(UNDEFINED, NO_ACCESS, false), None);
        assert_eq!(state_of(12, 0, false), None, "a video layout");
    }

    #[test]
    fn a_state_the_pcs_texture_or_list_cannot_have_is_left_out_or_refused() {
        // (flags: 1 render target, 2 depth, 4 unordered access, 8 no shader resource)
        assert_eq!(legal(RENDER_TARGET, 0x1, true), Some(RENDER_TARGET));
        assert_eq!(legal(RENDER_TARGET, 0x0, true), None, "the PC's copy is no render target");
        assert_eq!(legal(UNORDERED_ACCESS, 0x1, true), None);
        assert_eq!(legal(DEPTH_WRITE, 0x1, true), None);
        assert_eq!(legal(SHADER_RESOURCE | DEPTH_READ, 0x0, true), Some(SHADER_RESOURCE), "not a depth buffer");
        assert_eq!(legal(SHADER_RESOURCE | DEPTH_READ, 0x2 | 0x8, true), Some(DEPTH_READ), "a depth buffer no shader reads");
        assert_eq!(legal(0, 0x0, true), Some(0));
        // a compute list
        assert_eq!(legal(SHADER_RESOURCE, 0x0, false), Some(0x40));
        assert_eq!(legal(UNORDERED_ACCESS, 0x4, false), Some(UNORDERED_ACCESS));
        assert_eq!(legal(RENDER_TARGET, 0x1, false), None);
        assert_eq!(legal(DEPTH_READ | SHADER_RESOURCE, 0x2, false), None);
        assert_eq!(legal(COPY_SOURCE | SHADER_RESOURCE, 0x0, false), Some(COPY_SOURCE | 0x40));
    }

    #[test]
    fn a_range_names_its_subresources_or_all_of_them() {
        // an index
        assert_eq!(subresources([u32::MAX, 0, 0, 0, 0, 0], 4, 2, 1), None);
        assert_eq!(subresources([5, 0, 0, 0, 0, 0], 4, 2, 1), Some(vec![5]));
        assert_eq!(subresources([0, 0, 0, 0, 0, 0], 1, 1, 1), None, "the one there is");
        // a range: all of it, a level of each slice, the stencil plane
        assert_eq!(subresources([0, 4, 0, 2, 0, 1], 4, 2, 1), None);
        assert_eq!(subresources([1, 1, 0, 2, 0, 1], 4, 2, 1), Some(vec![1, 5]));
        assert_eq!(subresources([0, 1, 0, 1, 1, 1], 1, 1, 2), Some(vec![1]));
        assert_eq!(subresources([0, 1, 0, 0, 0, 0], 1, 1, 1), None, "no slices and planes named: one of each");
        // past the texture: what there is of it
        assert_eq!(subresources([2, 8, 0, 1, 0, 1], 4, 1, 1), Some(vec![2, 3]));
        // more than are named one by one: all
        assert_eq!(subresources([0, 1, 0, 100, 0, 1], 2, 100, 1), None);
    }

    fn t(res: usize, layout: (u32, u32), access: (u32, u32)) -> Texture {
        Texture { sync: (0, 0), access, layout, res, range: [u32::MAX, 0, 0, 0, 0, 0] }
    }

    fn run(b: &[Texture], last: &mut std::collections::HashMap<usize, u32>) -> Vec<(usize, u32, u32, u32, u32)> {
        let known = last.clone();
        let mut moved = Vec::new();
        let out = transitions_of(b, &mut |res| (res != 0xBAD).then_some((res == 0xD, 1, 1, if res == 0xD { 2 } else { 1 })), &mut |res| known.get(&res).copied(), &mut |res, s| moved.push((res, s)));
        last.extend(moved);
        out.iter().map(|w| ((w[2] as u64 | (w[3] as u64) << 32) as usize, w[4], w[5], w[6], w[1])).collect()
    }

    #[test]
    fn texture_barriers_become_transitions_between_their_layouts_states() {
        let mut last = Default::default();
        // render target → sampled → render target
        assert_eq!(run(&[t(0xA, (2, 6), (0x8, 0x80))], &mut last), [(0xA, ALL, RENDER_TARGET, SHADER_RESOURCE, 0)]);
        assert_eq!(run(&[t(0xA, (6, 2), (0x80, 0x8))], &mut last), [(0xA, ALL, SHADER_RESOURCE, RENDER_TARGET, 0)]);
        // a depth buffer into unordered access and out of it
        assert_eq!(run(&[t(0xD, (4, 3), (0x20, 0x10)), t(0xD, (3, 6), (0x10, 0x80))], &mut last), [(0xD, ALL, DEPTH_WRITE, UNORDERED_ACCESS, 0), (0xD, ALL, UNORDERED_ACCESS, SHADER_RESOURCE, 0)]);
        // only the access changes: nothing to move
        assert_eq!(run(&[t(0xA, (3, 3), (0x10, 0x10)), t(0xA, (6, 21), (0x80, 0x80))], &mut last), []);
        // not a texture XCL can ask, and no resource
        assert_eq!(run(&[t(0xBAD, (2, 6), (0x8, 0x80)), t(0, (2, 6), (0x8, 0x80))], &mut last), []);
    }

    #[test]
    fn the_undefined_layout_moves_nothing_and_is_left_from_where_the_texture_was() {
        let mut last = Default::default();
        // never moved here: where it goes is noted, nothing is sent
        assert_eq!(run(&[t(0xB, (UNDEFINED, 2), (NO_ACCESS, 0x8))], &mut last), []);
        assert_eq!(last.get(&0xB), Some(&RENDER_TARGET));
        // to the undefined layout: it stays a render target
        assert_eq!(run(&[t(0xB, (2, UNDEFINED), (0x8, NO_ACCESS))], &mut last), []);
        assert_eq!(last.get(&0xB), Some(&RENDER_TARGET));
        // from it into unordered access: from the render target it still is
        assert_eq!(run(&[t(0xB, (UNDEFINED, 3), (NO_ACCESS, 0x10))], &mut last), [(0xB, ALL, RENDER_TARGET, UNORDERED_ACCESS, 0)]);
        // and where it already is: nothing
        assert_eq!(run(&[t(0xB, (UNDEFINED, 3), (NO_ACCESS, 0x10))], &mut last), []);
        // a texture first seen leaving for the undefined layout is where it says it was
        assert_eq!(run(&[t(0xC, (6, UNDEFINED), (0x80, NO_ACCESS))], &mut last), []);
        assert_eq!(last.get(&0xC), Some(&SHADER_RESOURCE));
    }

    #[test]
    fn the_halves_of_a_split_barrier_are_the_pcs_begin_and_end() {
        let mut last = Default::default();
        let mut first = t(0xA, (2, 6), (0x8, 0x80));
        first.sync = (0x4, SYNC_SPLIT);
        let mut second = first;
        second.sync = (SYNC_SPLIT, 0x8);
        assert_eq!(run(&[first], &mut last), [(0xA, ALL, RENDER_TARGET, SHADER_RESOURCE, BEGIN_ONLY)]);
        assert_eq!(run(&[second], &mut last), [(0xA, ALL, RENDER_TARGET, SHADER_RESOURCE, END_ONLY)]);
    }

    #[test]
    fn a_ranged_barrier_names_each_subresource() {
        let mut last = Default::default();
        let mut b = t(0xD, (4, 6), (0x20, 0x80));
        b.range = [0, 1, 0, 1, 1, 1];
        assert_eq!(run(&[b], &mut last), [(0xD, 1, DEPTH_WRITE, SHADER_RESOURCE, 0)], "the stencil plane of a one-level depth buffer");
    }
}
