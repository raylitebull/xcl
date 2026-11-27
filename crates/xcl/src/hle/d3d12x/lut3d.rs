//! `XCL_DUMP_TARGETS` (`targets`): 3D textures – a title's colour-grading lookup tables are –
//! followed through XCL. With the switch on, a line when one is made (whichever creation path:
//! `lut<n>` names it from then on), when XCL sees it written (a copy into it, a render target view
//! of it made, drawn into or cleared, an unordered access view of it made – what a shader writes
//! through one is not seen –, an upload from the title's memory, WriteToSubresource) and when a
//! shader resource view of it goes into a console descriptor slot (its T# index there is what a
//! translated shader takes from `ResourceDescriptorHeap`). At a dumped frame each live one of at
//! most 64 on every side is read back in the state the title's submitted work left it in (its
//! transitions, as `title_textures` notes them, applied at each submit): the texels of its 8
//! corners and centre and each channel's least and greatest over mip 0, and its first, middle
//! and last depth slices as `lut<n>_<w>x<h>x<d>_f<format>_slice<k>.bmp` (channels clamped to
//! 0..1, alpha left out). One whose state is not known (copied into from COMMON: promoted to
//! COPY_DEST, which a texture does not decay from) is logged as not read back.
//! Textures are kept by address (not AddRef'd) and forgotten at their last Release.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// sides of a texture read back at a dump, at most; and how many are read back
const SIDE_MOST: u64 = 64;
const DUMPED_MOST: usize = 256;
/// lines of writes logged per texture (then every power of two)
const WRITES_LOGGED: u64 = 12;

pub fn on() -> bool {
    super::targets::enabled()
}

/// Whether `created` is to be told of a resource made (`on`, or `XCL_CAPTURE_FRAME`).
pub fn created_on() -> bool {
    on() || super::capture::enabled()
}

struct Vol {
    n: u64,
    /// the state the title's submitted work leaves it in: one for all subresources, or one each
    states: Vec<D3D12_RESOURCE_STATES>,
    /// copied into while in COMMON with no transition after: implicitly COPY_DEST
    promoted: bool,
    writes: u64,
}

static VOLS: Mutex<BTreeMap<usize, Vol>> = Mutex::new(BTreeMap::new());
static NEXT: AtomicU64 = AtomicU64::new(0);
/// PC list → the transitions of 3D textures it recorded since it was made or reset
static NOTES: Mutex<BTreeMap<usize, Vec<(usize, u32, D3D12_RESOURCE_STATES)>>> = Mutex::new(BTreeMap::new());

unsafe fn desc_of(res: usize) -> Option<D3D12_RESOURCE_DESC> {
    let p = res as *mut c_void;
    ID3D12Resource::from_raw_borrowed(&p).map(|r| r.GetDesc())
}

/// Texture `res` was made by `how` (from the console description `console`, at the title's
/// address `va`, kept in step with that memory or not) in `state`: noted if it is a 3D one.
pub unsafe fn created(how: &str, res: *mut c_void, console: Option<&[u8; 0x38]>, va: Option<usize>, kept: bool, state: u32) {
    super::capture::created(how, res, console, va, kept);
    if !on() || res.is_null() {
        return;
    }
    let Some(d) = desc_of(res as usize).filter(|d| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) else { return };
    let u = |x: &[u8; 0x38], o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let console_text = console.map_or(String::new(), |x| format!(" (console format {}, console flags {:#x}, layout {:#x})", u(x, 0x20), u(x, 0x30), u(x, 0x2C)));
    let flags = d.Flags.0;
    let (n, known) = {
        let mut g = VOLS.lock().unwrap();
        match g.get(&(res as usize)) {
            Some(v) => (v.n, true),
            None => {
                let n = NEXT.fetch_add(1, Ordering::Relaxed);
                g.insert(res as usize, Vol { n, states: vec![D3D12_RESOURCE_STATES(state as i32)], promoted: false, writes: 0 });
                (n, false)
            }
        }
    };
    let at = va.map_or(String::new(), |a| format!(", the title's memory at {a:#x}"));
    match known {
        true => crate::xlog_info!("D3D12.X: 3D texture lut{n} {res:p} is also {how}{console_text}: {}{at}", if kept { "kept in step with the title's memory" } else { "not kept in step with the title's memory" }),
        false => crate::xlog_info!(
            "D3D12.X: 3D texture lut{n} {res:p} made by {how}: {}x{}x{}, {} mips, PC format {}{console_text}, flags {flags:#x} (render target {}, unordered access {}), {}{at}, state {state:#x}",
            d.Width,
            d.Height,
            d.DepthOrArraySize,
            d.MipLevels,
            d.Format.0,
            flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 != 0,
            flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0,
            if kept { "kept in step with the title's memory (uploaded from it)" } else { "not kept in step with the title's memory (its contents are the GPU's)" }
        ),
    }
}

/// The PC texture went for good.
pub fn forget(res: usize) {
    if on() {
        VOLS.lock().unwrap().remove(&res);
    }
}

/// `res` (any resource) is written by `what` (`copy`: a copy, which promotes a texture in COMMON
/// to COPY_DEST): logged if it is a 3D texture (the first `WRITES_LOGGED` times, then every
/// power of two).
pub fn wrote(res: usize, copy: bool, what: impl FnOnce() -> String) {
    if !on() || res == 0 {
        return;
    }
    let (n, k) = {
        let mut g = VOLS.lock().unwrap();
        let Some(v) = g.get_mut(&res) else { return };
        v.writes += 1;
        if copy && v.states.len() == 1 && v.states[0] == D3D12_RESOURCE_STATE_COMMON {
            v.promoted = true;
        }
        (v.n, v.writes)
    };
    if k <= WRITES_LOGGED || k.is_power_of_two() {
        crate::xlog_info!("D3D12.X: 3D texture lut{n} {res:#x} written (#{k}): {}", what());
    }
}

/// A shader resource view `v` (None: the default one) of `res` went into console slot `cpu`, T#
/// index `index`.
pub fn viewed(res: usize, rd: &D3D12_RESOURCE_DESC, v: Option<&D3D12_SHADER_RESOURCE_VIEW_DESC>, index: u32, cpu: u64) {
    if !on() {
        return;
    }
    if rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        return slot_reused(index, cpu, res, || format!("a view of {res:#x} (dimension {}, {}x{}, format {})", rd.Dimension.0, rd.Width, rd.Height, rd.Format.0));
    }
    let Some(n) = VOLS.lock().unwrap().get(&res).map(|v| v.n) else { return };
    INDICES.lock().unwrap().insert(index, n);
    let view = match v {
        // SAFETY: the union member the view dimension names
        Some(v) if v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE3D => unsafe {
            let t = v.Anonymous.Texture3D;
            format!("view format {}, mips {} (+{}), min LOD clamp {}, component mapping {:#x}", v.Format.0, t.MostDetailedMip, t.MipLevels as i32, t.ResourceMinLODClamp, v.Shader4ComponentMapping)
        },
        Some(v) => format!("view dimension {} (not 3D), format {}", v.ViewDimension.0, v.Format.0),
        None => format!("the default view (format {}, all {} mips)", rd.Format.0, rd.MipLevels),
    };
    crate::xlog_info!("D3D12.X: 3D texture lut{n} {res:#x}: shader resource view in console slot {cpu:#x} (T# index {index}): {view}");
}

/// T# indices that hold a shader resource view of a 3D texture → its number
static INDICES: Mutex<BTreeMap<u32, u64>> = Mutex::new(BTreeMap::new());

/// Console slot `cpu` (T# index `index`) gets `what`, not a view of a 3D texture: logged when it
/// held one – a T# the title copied out of the slot before now names `what` on the PC, where the
/// console's copy still names the 3D texture's memory.
pub fn slot_reused(index: u32, cpu: u64, res: usize, what: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let same = |n: u64| VOLS.lock().unwrap().get(&res).is_some_and(|v| v.n == n);
    let Some(n) = INDICES.lock().unwrap().get(&index).copied().filter(|&n| !same(n)) else { return };
    INDICES.lock().unwrap().remove(&index);
    crate::xlog_info!("D3D12.X: 3D texture lut{n}: console slot {cpu:#x} (T# index {index}), which held a view of it, now holds {}", what());
}

/// The title recorded `transitions` (resource, subresource, state after) on PC list `list`.
pub fn note_transitions(list: usize, transitions: &[(usize, u32, D3D12_RESOURCE_STATES)]) {
    if !on() || transitions.is_empty() {
        return;
    }
    let ours: Vec<_> = {
        let g = VOLS.lock().unwrap();
        transitions.iter().filter(|t| g.contains_key(&t.0)).copied().collect()
    };
    if !ours.is_empty() {
        NOTES.lock().unwrap().entry(list).or_default().extend(ours);
    }
}

/// PC list `list` records anew.
pub fn list_reset(list: usize) {
    if on() {
        NOTES.lock().unwrap().remove(&list);
    }
}

/// PC lists `lists` were submitted in this order: their transitions apply.
pub fn submitted(lists: &[*mut c_void]) {
    if !on() {
        return;
    }
    let notes: Vec<_> = {
        let g = NOTES.lock().unwrap();
        if g.is_empty() { return }
        lists.iter().filter_map(|&l| g.get(&(l as usize))).flatten().copied().collect()
    };
    if notes.is_empty() {
        return;
    }
    let mut g = VOLS.lock().unwrap();
    for (res, sub, after) in notes {
        if let Some(v) = g.get_mut(&res) {
            let mips = unsafe { desc_of(res) }.map_or(1, |d| d.MipLevels.max(1) as usize);
            super::title_textures::apply_transition(&mut v.states, mips, sub, after);
            v.promoted = false;
        }
    }
}

/// The 3D textures to read back at this dump: (tag, texture, its desc, the state subresource 0
/// is in); the others are logged with why not.
pub unsafe fn to_dump() -> Vec<(String, ID3D12Resource, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES)> {
    let g = VOLS.lock().unwrap();
    crate::xlog_info!("D3D12.X: 3D texture dump: {} 3D textures live", g.len());
    let mut out = Vec::new();
    for (&res, v) in g.iter() {
        let p = res as *mut c_void;
        let Some(r) = ID3D12Resource::from_raw_borrowed(&p) else { continue };
        let d = r.GetDesc();
        let why = if d.Width > SIDE_MOST || d.Height as u64 > SIDE_MOST || d.DepthOrArraySize as u64 > SIDE_MOST {
            Some("larger than 64 on a side".to_string())
        } else if texel_floats(d.Format, &[0; 16]).is_none() {
            Some(format!("format {} is not decoded here", d.Format.0))
        } else if v.promoted {
            Some("its state is not known (copied into from COMMON, so in COPY_DEST)".into())
        } else if out.len() >= DUMPED_MOST {
            Some(format!("past the {DUMPED_MOST} read back"))
        } else {
            None
        };
        let state = v.states[0];
        match why {
            Some(w) => crate::xlog_info!("D3D12.X: 3D texture dump: lut{} {res:#x} {}x{}x{} format {} (state {:#x}): not read back – {w}", v.n, d.Width, d.Height, d.DepthOrArraySize, d.Format.0, state.0),
            None => out.push((format!("lut{}_{}x{}x{}_f{}", v.n, d.Width, d.Height, d.DepthOrArraySize, d.Format.0), r.clone(), d, state)),
        }
    }
    out
}

/// Whether a copy from a texture in `state` needs a transition to COPY_SOURCE first (COMMON is
/// promoted to it implicitly; a read state that has COPY_SOURCE in it is one already).
pub fn needs_transition(state: D3D12_RESOURCE_STATES) -> bool {
    state != D3D12_RESOURCE_STATE_COMMON && state.0 & D3D12_RESOURCE_STATE_COPY_SOURCE.0 == 0
}

/// A texel of `format` (its first `targets::texel_bytes` bytes of `t`) as four floats, as the
/// texture stores it (sRGB not linearised; missing channels 0, missing alpha 1). None: a format
/// not decoded here (block-compressed ones among them).
pub fn texel_floats(format: DXGI_FORMAT, t: &[u8]) -> Option<[f32; 4]> {
    use super::present::{half_float, small_float};
    if t.len() < super::targets::texel_bytes(format) {
        return None;
    }
    let u16_at = |i: usize| u16::from_le_bytes([t[i], t[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([t[i], t[i + 1], t[i + 2], t[i + 3]]);
    let h = |i: usize| half_float(u16_at(i));
    let f = |i: usize| f32::from_bits(u32_at(i));
    let un8 = |i: usize| t[i] as f32 / 255.0;
    let un16 = |i: usize| u16_at(i) as f32 / 65535.0;
    Some(match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_R8G8B8A8_TYPELESS => [un8(0), un8(1), un8(2), un8(3)],
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_TYPELESS => [un8(2), un8(1), un8(0), un8(3)],
        DXGI_FORMAT_B8G8R8X8_UNORM | DXGI_FORMAT_B8G8R8X8_UNORM_SRGB => [un8(2), un8(1), un8(0), 1.0],
        DXGI_FORMAT_R8G8B8A8_SNORM => {
            let s = |i: usize| (t[i] as i8 as f32 / 127.0).max(-1.0);
            [s(0), s(1), s(2), s(3)]
        }
        DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_TYPELESS => {
            let v = u32_at(0);
            [(v & 0x3FF) as f32 / 1023.0, (v >> 10 & 0x3FF) as f32 / 1023.0, (v >> 20 & 0x3FF) as f32 / 1023.0, (v >> 30) as f32 / 3.0]
        }
        DXGI_FORMAT_R11G11B10_FLOAT => {
            let v = u32_at(0);
            [small_float(v & 0x7FF, 6), small_float(v >> 11 & 0x7FF, 6), small_float(v >> 22, 5), 1.0]
        }
        DXGI_FORMAT_R9G9B9E5_SHAREDEXP => {
            let v = u32_at(0);
            let scale = 2f32.powi((v >> 27) as i32 - 15 - 9);
            [(v & 0x1FF) as f32 * scale, (v >> 9 & 0x1FF) as f32 * scale, (v >> 18 & 0x1FF) as f32 * scale, 1.0]
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS => [h(0), h(2), h(4), h(6)],
        DXGI_FORMAT_R16G16B16A16_UNORM => [un16(0), un16(2), un16(4), un16(6)],
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_TYPELESS => [f(0), f(4), f(8), f(12)],
        DXGI_FORMAT_R32G32_FLOAT => [f(0), f(4), 0.0, 1.0],
        DXGI_FORMAT_R16G16_FLOAT => [h(0), h(2), 0.0, 1.0],
        DXGI_FORMAT_R16G16_UNORM => [un16(0), un16(2), 0.0, 1.0],
        DXGI_FORMAT_R8G8_UNORM => [un8(0), un8(1), 0.0, 1.0],
        DXGI_FORMAT_R32_FLOAT => [f(0), 0.0, 0.0, 1.0],
        DXGI_FORMAT_R16_FLOAT => [h(0), 0.0, 0.0, 1.0],
        DXGI_FORMAT_R16_UNORM => [un16(0), 0.0, 0.0, 1.0],
        DXGI_FORMAT_R8_UNORM => [un8(0), 0.0, 0.0, 1.0],
        _ => return None,
    })
}

/// The 8 corners of a `w` × `h` × `d` volume and its centre, in that order (x fastest).
fn probe_points(w: usize, h: usize, d: usize) -> [(usize, usize, usize); 9] {
    let (x, y, z) = (w.max(1) - 1, h.max(1) - 1, d.max(1) - 1);
    [(0, 0, 0), (x, 0, 0), (0, y, 0), (x, y, 0), (0, 0, z), (x, 0, z), (0, y, z), (x, y, z), (w / 2, h / 2, d / 2)]
}

/// Byte offset of texel (x, y, z) in a footprint of `h` rows of `pitch` bytes per depth slice.
fn texel_offset(x: usize, y: usize, z: usize, h: usize, pitch: usize, bytes: usize) -> usize {
    (z * h + y) * pitch + x * bytes
}

/// Least and greatest of each channel over `texels`.
fn channel_range(texels: impl Iterator<Item = [f32; 4]>) -> ([f32; 4], [f32; 4]) {
    let (mut lo, mut hi) = ([f32::INFINITY; 4], [f32::NEG_INFINITY; 4]);
    for t in texels {
        for c in 0..4 {
            lo[c] = lo[c].min(t[c]);
            hi[c] = hi[c].max(t[c]);
        }
    }
    (lo, hi)
}

fn text(t: [f32; 4]) -> String {
    format!("({:.4}, {:.4}, {:.4}, {:.4})", t[0], t[1], t[2], t[3])
}

/// A 24-bit BMP of `w` × `h` colours (rows top first).
fn bmp(w: usize, h: usize, rgb: &[[u8; 3]]) -> Vec<u8> {
    let row = (w * 3).div_ceil(4) * 4;
    let size = (54 + row * h) as u32;
    let mut b = Vec::with_capacity(size as usize);
    b.extend_from_slice(b"BM");
    for v in [size, 0, 54, 40, w as u32, h as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&24u16.to_le_bytes());
    b.extend_from_slice(&[0; 24]);
    for y in (0..h).rev() {
        let start = b.len();
        for c in &rgb[y * w..(y + 1) * w] {
            b.extend_from_slice(&[c[2], c[1], c[0]]);
        }
        b.resize(start + row, 0);
    }
    b
}

/// A dump `tag` (`to_dump`) read back into `rb` at `fp` (mip 0, all its depth slices): its texel
/// values logged and three depth slices written as pictures.
pub unsafe fn log_values(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let f = &fp.Footprint;
    let (w, h, d, pitch) = (f.Width as usize, f.Height as usize, f.Depth as usize, f.RowPitch as usize);
    let bytes = super::targets::texel_bytes(f.Format);
    let len = (rb.GetDesc().Width as usize).saturating_sub(fp.Offset as usize);
    let data = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize), len);
    let at = |x: usize, y: usize, z: usize| data.get(texel_offset(x, y, z, h, pitch, bytes)..).and_then(|t| texel_floats(f.Format, t));
    let points: Vec<String> = probe_points(w, h, d).iter().map(|&(x, y, z)| format!("({x},{y},{z})={}", at(x, y, z).map_or("?".into(), text))).collect();
    let (lo, hi) = channel_range((0..d).flat_map(|z| (0..h).flat_map(move |y| (0..w).map(move |x| (x, y, z)))).filter_map(|(x, y, z)| at(x, y, z)));
    crate::xlog_info!("D3D12.X: 3D texture dump: {tag}: corners and centre (x,y,z)=(r,g,b,a): {}", points.join(" "));
    crate::xlog_info!("D3D12.X: 3D texture dump: {tag}: over mip 0, least {} greatest {}", text(lo), text(hi));
    let dir = crate::state::boot().map(|b| b.logs_dir());
    let mut slices = vec![0, d / 2, d.max(1) - 1];
    slices.dedup();
    for k in slices {
        let rgb: Vec<[u8; 3]> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| at(x, y, k).map_or([0; 3], |t| [0, 1, 2].map(|c| if t[c].is_nan() { 255 } else { (t[c].clamp(0.0, 1.0) * 255.0).round() as u8 })))
            .collect();
        if let Some(dir) = &dir {
            let path = dir.join(format!("{tag}_slice{k}.bmp"));
            let _ = std::fs::write(&path, bmp(w, h, &rgb));
            crate::xlog_info!("D3D12.X: 3D texture dump: {tag}: depth slice {k} written to {}", path.display());
        }
    }
    rb.Unmap(0, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Texels of the formats a lookup table has, as the floats a shader would read.
    #[test]
    fn lut_texels_decode_to_floats() {
        assert_eq!(texel_floats(DXGI_FORMAT_R8G8B8A8_UNORM, &[0, 255, 51, 255]), Some([0.0, 1.0, 0.2, 1.0]));
        assert_eq!(texel_floats(DXGI_FORMAT_B8G8R8A8_UNORM, &[0, 255, 51, 255]), Some([0.2, 1.0, 0.0, 1.0]));
        // R10G10B10A2: r 1023 in the low bits, g 0, b 0, a 3
        assert_eq!(texel_floats(DXGI_FORMAT_R10G10B10A2_UNORM, &(0x3FFu32 | 3 << 30).to_le_bytes()), Some([1.0, 0.0, 0.0, 1.0]));
        // R11G11B10: 1.0 = exponent 15, no mantissa, in each of r (bits 0-10) and b (22-31)
        let one11 = 15u32 << 6;
        let one10 = 15u32 << 5;
        assert_eq!(texel_floats(DXGI_FORMAT_R11G11B10_FLOAT, &(one11 | one10 << 22).to_le_bytes()), Some([1.0, 0.0, 1.0, 1.0]));
        // half floats 1.0, 0.5, 0, -2
        let hs: Vec<u8> = [0x3C00u16, 0x3800, 0, 0xC000].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(texel_floats(DXGI_FORMAT_R16G16B16A16_FLOAT, &hs), Some([1.0, 0.5, 0.0, -2.0]));
        let fs: Vec<u8> = [0.25f32, 1.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(texel_floats(DXGI_FORMAT_R32G32B32A32_FLOAT, &fs), Some([0.25, 1.0, 0.0, 1.0]));
        // R9G9B9E5: mantissa 256 with exponent 16: 256 * 2^(16 - 24) = 1.0 for r
        assert_eq!(texel_floats(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, &(256u32 | 16 << 27).to_le_bytes()), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(texel_floats(DXGI_FORMAT_BC1_UNORM, &[0; 16]), None);
        assert_eq!(texel_floats(DXGI_FORMAT_R16G16B16A16_FLOAT, &[0; 4]), None, "too few bytes");
    }

    /// The corners and centre of a volume, where a texel lies in a footprint, and the channels'
    /// ranges; which states a copy needs a transition from.
    #[test]
    fn lut_probes_and_ranges() {
        let p = probe_points(32, 16, 8);
        assert_eq!(p[0], (0, 0, 0));
        assert_eq!(p[7], (31, 15, 7));
        assert_eq!(p[6], (0, 15, 7));
        assert_eq!(p[8], (16, 8, 4));
        assert_eq!(probe_points(1, 1, 1), [(0, 0, 0); 9]);
        // 32 wide RGBA16F: 256-byte rows, 16 rows a slice
        assert_eq!(texel_offset(3, 2, 1, 16, 256, 8), (16 + 2) * 256 + 24);
        let (lo, hi) = channel_range([[0.0, 1.0, 1.0, 1.0], [0.5, 0.25, 1.0, 1.0]].into_iter());
        assert_eq!((lo, hi), ([0.0, 0.25, 1.0, 1.0], [0.5, 1.0, 1.0, 1.0]));
        assert!(!needs_transition(D3D12_RESOURCE_STATE_COMMON));
        assert!(!needs_transition(D3D12_RESOURCE_STATE_GENERIC_READ));
        assert!(needs_transition(D3D12_RESOURCE_STATES(0xC0)));
        assert!(needs_transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        // a 2 x 1 picture: a row padded to 8 bytes, bottom row first
        let b = bmp(2, 1, &[[1, 2, 3], [4, 5, 6]]);
        assert_eq!(b.len(), 54 + 8);
        assert_eq!(&b[54..], &[3, 2, 1, 6, 5, 4, 0, 0]);
    }
}
