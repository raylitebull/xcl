//! Sky-following stand-in ambient lighting when ray-traced indirect illumination is gated off.
//!
//! When `XCL_RT_GATE` holds back ray-traced indirect lighting passes, screen-space probe resolve
//! (`CS_ScreenProbes_TemporalResolve`) receives no ray-traced bounce irradiance (leaving foliage,
//! cloth, and indoor shadows dark or corrupted) and reflection history (`CS_UpdateHistory`) can
//! retain unoccluded sky glare indoors.
//!
//! Controlled by `XCL_AMBIENT`:
//! * `auto` (default): active only while `XCL_RT_GATE` is holding back BVH traversal.
//! * `sky` / `1` / `on`: always active.
//! * `0` / `off` / `none`: disabled.
//!
//! Tunables: `XCL_AMBIENT_GAIN`, `XCL_AMBIENT_POWER`, `XCL_AMBIENT_LEAST`, `XCL_AMBIENT_AO`,
//! `XCL_AMBIENT_REFLECT`, `XCL_AMBIENT_SETTLE_MS`, `XCL_AMBIENT_AIR`.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

use windows::core::{Interface, PCSTR};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::console_list::{Layout, Param};
use super::{list_x, present, real};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    Auto,
    Sky,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    mode: Mode,
    gain: f32,
    power: f32,
    least: f32,
    ao: f32,
    reflect: f32,
    settle_ms: f32,
    air: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Auto,
            gain: 0.22,
            power: 0.65,
            least: 0.02,
            ao: 0.55,
            reflect: 0.50,
            settle_ms: 250.0,
            air: 0.0,
        }
    }
}

fn parse_mode(raw: Option<&str>) -> Mode {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("0" | "off" | "none" | "false") => Mode::Off,
        Some("sky" | "1" | "on" | "true") => Mode::Sky,
        _ => Mode::Auto,
    }
}

fn env_f32(name: &str, default: f32, lo: f32, hi: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .map_or(default, |v| v.clamp(lo, hi))
}

fn config() -> &'static Config {
    static CFG: OnceLock<Config> = OnceLock::new();
    CFG.get_or_init(|| {
        let def = Config::default();
        let raw = std::env::var("XCL_AMBIENT").ok();
        Config {
            mode: parse_mode(raw.as_deref()),
            gain: env_f32("XCL_AMBIENT_GAIN", def.gain, 0.0, 8.0),
            power: env_f32("XCL_AMBIENT_POWER", def.power, 0.1, 4.0),
            least: env_f32("XCL_AMBIENT_LEAST", def.least, 0.0, 2.0),
            ao: env_f32("XCL_AMBIENT_AO", def.ao, 0.0, 1.0),
            reflect: env_f32("XCL_AMBIENT_REFLECT", def.reflect, 0.05, 4.0),
            settle_ms: env_f32("XCL_AMBIENT_SETTLE_MS", def.settle_ms, 0.0, 10_000.0),
            air: env_f32("XCL_AMBIENT_AIR", def.air, 0.0, 16.0),
        }
    })
}

/// Time-based EMA smoothing factor `1 - exp(-dt / tau)` for sky-dome ambient settling.
/// Snaps immediately (`1.0`) on the first captured sample or when `settle_ms <= 0`.
fn settle_alpha(dt_s: f32, settle_ms: f32, first_sample: bool) -> f32 {
    if first_sample || settle_ms <= 0.0 {
        return 1.0;
    }
    let dt = dt_s.clamp(0.0, 2.0);
    let tau = (settle_ms * 1e-3).max(1e-4);
    (1.0 - (-dt / tau).exp()).clamp(0.0, 1.0)
}

fn monotonic_us() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let base = EPOCH.get_or_init(Instant::now);
    (base.elapsed().as_micros() as u64).saturating_add(1)
}

/// Whether sky-following stand-in ambient tracking is enabled (`XCL_AMBIENT != 0/off`).
pub fn enabled() -> bool {
    config().mode != Mode::Off
}

/// Whether stand-in ambient passes should currently execute.
pub fn active() -> bool {
    match config().mode {
        Mode::Off => false,
        Mode::Sky => true,
        Mode::Auto => rays_find_nothing(real::bvh_traversal_allowed(), crate::gpu::dxil::kernel::rt_all_miss()),
    }
}

/// The title's rays bring no light: traversal is held back, or runs with every ray missing.
fn rays_find_nothing(traversal_allowed: bool, every_ray_misses: bool) -> bool {
    !traversal_allowed || every_ray_misses
}

/// Role of a console shader identified from FSSM/container metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PassRole {
    SkyDome,
    TemporalResolve,
    ReflectionHistory,
    InscatteringVolume,
}

fn contains_ascii_ci(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle))
}

fn classify_ps(blob: &[u8]) -> Option<PassRole> {
    if contains_ascii_ci(blob, b"ps_sky_paraboloid_reflection") {
        Some(PassRole::SkyDome)
    } else {
        None
    }
}

fn classify_cs(blob: &[u8]) -> Option<PassRole> {
    if contains_ascii_ci(blob, b"CS_ScreenProbes_TemporalResolve") {
        Some(PassRole::TemporalResolve)
    } else if contains_ascii_ci(blob, b"CS_UpdateHistory") {
        Some(PassRole::ReflectionHistory)
    } else if contains_ascii_ci(blob, b"CS_AccumulateLocalLightInscatteringVolume") {
        Some(PassRole::InscatteringVolume)
    } else {
        None
    }
}

static PSO_ROLES: RwLock<Option<HashMap<usize, PassRole>>> = RwLock::new(None);
static HAS_SKY_PSO: AtomicBool = AtomicBool::new(false);
static HAS_CS_PSO: AtomicBool = AtomicBool::new(false);

pub(super) fn note_graphics_pso(pso: usize, ps: Option<&[u8]>) {
    if pso == 0 || !enabled() {
        return;
    }
    if let Some(role) = ps.and_then(classify_ps) {
        PSO_ROLES.write().unwrap().get_or_insert_with(HashMap::new).insert(pso, role);
        HAS_SKY_PSO.store(true, Ordering::Release);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_info!("D3D12.X: ambient: tracked sky paraboloid PSO {pso:#x}");
        }
    }
}

/// Whether a compute shader blob belongs to a pass handled by stand-in ambient lighting
/// (`CS_ScreenProbes_TemporalResolve` / `CS_UpdateHistory` /
/// `CS_AccumulateLocalLightInscatteringVolume`).
pub(super) fn is_ambient_compute(blob: &[u8]) -> bool {
    classify_cs(blob).is_some()
}

pub(super) fn note_compute_pso(pso: usize, cs: &[u8]) {
    if pso == 0 || !enabled() {
        return;
    }
    if let Some(role) = classify_cs(cs) {
        PSO_ROLES.write().unwrap().get_or_insert_with(HashMap::new).insert(pso, role);
        HAS_CS_PSO.store(true, Ordering::Release);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            crate::xlog_info!("D3D12.X: ambient: tracked {role:?} compute PSO {pso:#x}");
        }
    }
}

#[inline]
pub(super) fn has_sky_psos() -> bool {
    HAS_SKY_PSO.load(Ordering::Relaxed)
}

#[inline]
pub(super) fn has_compute_psos() -> bool {
    HAS_CS_PSO.load(Ordering::Relaxed)
}

fn role_of(pso: usize) -> Option<PassRole> {
    if pso == 0 {
        return None;
    }
    PSO_ROLES.read().unwrap().as_ref()?.get(&pso).copied()
}

/// Descriptor-heap view metadata indexed by BOTH console slot UMA address (`slot.at`, globally
/// unique across all shader-visible and CPU-only heaps) and `GLOBAL` heap descriptor index
/// (`slot.index` when hosted in `GLOBAL`) so both `CopyDescriptors*` and inline title `memcpy`
/// resolve without CPU-only heap index collisions.
#[derive(Clone, Copy, Debug)]
struct ViewEntry {
    res: usize,
    fmt: i32,
    is_uav: bool,
}

#[derive(Default)]
struct ViewSlots {
    by_addr: HashMap<usize, ViewEntry>,
    by_idx: HashMap<u32, ViewEntry>,
}

static VIEW_SLOTS: RwLock<Option<ViewSlots>> = RwLock::new(None);
static HAS_VIEW_SLOTS: AtomicBool = AtomicBool::new(false);

fn is_ambient_uav_format(fmt: i32) -> bool {
    // R16G16B16A16_FLOAT (10), R11G11B10_FLOAT (26), R32G32B32A32_FLOAT (2), R32_UINT (42, R9G9B9E5 packed)
    matches!(fmt, 10 | 26 | 2 | 42)
}

fn is_depth_srv_format(fmt: i32) -> bool {
    // R32_FLOAT (41), R24_UNORM_X8_TYPELESS (46), R32_FLOAT_X8X24_TYPELESS (21), R16_UNORM (56)
    matches!(fmt, 41 | 46 | 21 | 56)
}

fn is_depth_resource_format(fmt: i32) -> bool {
    matches!(fmt, 19 | 20 | 39 | 40 | 44 | 45 | 53 | 55)
}

/// Record a 2D texture UAV or depth SRV written at console slot address `slot_at` (and optional
/// `GLOBAL` descriptor index `global_index`, 0 when on a CPU-only / unhosted heap).
pub(super) fn note_descriptor(slot_at: usize, global_index: u32, res: usize, fmt: i32, res_fmt: i32, is_uav: bool) {
    if !enabled() || (slot_at == 0 && global_index == 0) {
        return;
    }
    if res == 0 {
        if HAS_VIEW_SLOTS.load(Ordering::Relaxed) {
            if let Some(s) = VIEW_SLOTS.write().unwrap().as_mut() {
                if slot_at != 0 {
                    s.by_addr.remove(&slot_at);
                }
                if global_index != 0 {
                    s.by_idx.remove(&global_index);
                }
            }
        }
        return;
    }
    let keep = if is_uav {
        is_ambient_uav_format(fmt)
    } else {
        is_depth_srv_format(fmt) && is_depth_resource_format(res_fmt)
    };
    let mut g = VIEW_SLOTS.write().unwrap();
    if keep {
        let s = g.get_or_insert_with(ViewSlots::default);
        let ve = ViewEntry { res, fmt, is_uav };
        if slot_at != 0 {
            s.by_addr.insert(slot_at, ve);
        }
        if global_index != 0 {
            s.by_idx.insert(global_index, ve);
        }
        HAS_VIEW_SLOTS.store(true, Ordering::Release);
    } else if let Some(s) = g.as_mut() {
        if slot_at != 0 {
            s.by_addr.remove(&slot_at);
        }
        if global_index != 0 {
            s.by_idx.remove(&global_index);
        }
    }
}

/// Evict any cached descriptor entries referencing a destroyed resource so stale raw pointers
/// are never dereferenced during dispatch lookup.
pub(super) fn forget_resource(res: usize) {
    if res == 0 || !HAS_VIEW_SLOTS.load(Ordering::Relaxed) {
        return;
    }
    if let Some(s) = VIEW_SLOTS.write().unwrap().as_mut() {
        s.by_addr.retain(|_, v| v.res != res);
        s.by_idx.retain(|_, v| v.res != res);
    }
}

/// Propagate view metadata when `CopyDescriptors*` copies from `src_at` (with pre-copy T# word 0
/// `src_global_index`) to `dst_at` (with destination `GLOBAL` index `dst_global_index`, 0 if unhosted).
pub(super) fn copy_descriptor(src_at: usize, src_global_index: u32, dst_at: usize, dst_global_index: u32) {
    if !enabled() || !HAS_VIEW_SLOTS.load(Ordering::Relaxed) || (src_at == dst_at && src_global_index == dst_global_index) {
        return;
    }
    let entry = VIEW_SLOTS.read().unwrap().as_ref().and_then(|s| {
        s.by_addr
            .get(&src_at)
            .or_else(|| (src_global_index != 0).then(|| s.by_idx.get(&src_global_index)).flatten())
            .copied()
    });
    let mut g = VIEW_SLOTS.write().unwrap();
    match entry {
        Some(v) => {
            let s = g.get_or_insert_with(ViewSlots::default);
            if dst_at != 0 {
                s.by_addr.insert(dst_at, v);
            }
            if dst_global_index != 0 {
                s.by_idx.insert(dst_global_index, v);
            }
        }
        None => {
            if let Some(s) = g.as_mut() {
                if dst_at != 0 {
                    s.by_addr.remove(&dst_at);
                }
                if dst_global_index != 0 {
                    s.by_idx.remove(&dst_global_index);
                }
            }
        }
    }
}

fn is_sky_dome_target(rd: &D3D12_RESOURCE_DESC) -> bool {
    rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && rd.Width == 128
        && rd.Height == 128
        && matches!(rd.Format, DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_TYPELESS)
        && rd.SampleDesc.Count == 1
}

/// Match a screen-space resolve/reflection output UAV against dispatch grid dimensions, returning
/// a lower-is-better score when compatible (`None` if incompatible).
/// `TemporalResolve` writes diffuse screen-probe light as packed `R9G9B9E5` in `R32_UINT` (`fmt 42`);
/// restricting `TemporalResolve` to `R32_UINT` avoids accidentally overwriting a `R11G11B10_FLOAT`
/// (`fmt 26`) HDR scene color buffer bound in the same descriptor heap.
fn score_dispatch_target_for_role(role: PassRole, rd: &D3D12_RESOURCE_DESC, view_fmt: i32, groups: [u32; 3]) -> Option<u64> {
    if role == PassRole::TemporalResolve && view_fmt != 42 {
        return None;
    }
    if role == PassRole::ReflectionHistory && view_fmt == 42 {
        return None;
    }
    score_dispatch_target(rd, view_fmt, groups)
}

fn score_dispatch_target(rd: &D3D12_RESOURCE_DESC, view_fmt: i32, groups: [u32; 3]) -> Option<u64> {
    if rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || rd.DepthOrArraySize != 1
        || rd.SampleDesc.Count != 1
        || rd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
        || !is_ambient_uav_format(view_fmt)
    {
        return None;
    }
    let w = rd.Width as u32;
    let h = rd.Height;
    if w < 64 || h < 64 {
        return None;
    }
    // Indirect or 1D tiled dispatch (`groups[0] == 0` or `groups[1] <= 1`): match screen-space 2D
    // float UAV targets (>= 128x72), preferring larger resolution.
    if groups[0] == 0 || groups[1] <= 1 {
        if w < 128 || h < 72 {
            return None;
        }
        return Some(u64::MAX - (w as u64).saturating_mul(h as u64));
    }
    let max_w = groups[0].saturating_mul(64);
    let max_h = groups[1].saturating_mul(64);
    let min_w = (groups[0] / 2).max(64);
    let min_h = (groups[1] / 2).max(64);
    if w < min_w || w > max_w || h < min_h || h > max_h {
        return None;
    }
    // Console compute shaders commonly use [8, 8, 1] or Wave32 [8, 4, 1] thread groups
    // (e.g. [320, 360, 1] -> 2560x1440 at 8x4 or 1280x720 at 4x2); score by closest fit.
    let d88 = (w as i64 - groups[0].saturating_mul(8) as i64).unsigned_abs()
        + (h as i64 - groups[1].saturating_mul(8) as i64).unsigned_abs();
    let d84 = (w as i64 - groups[0].saturating_mul(8) as i64).unsigned_abs()
        + (h as i64 - groups[1].saturating_mul(4) as i64).unsigned_abs();
    let d42 = (w as i64 - groups[0].saturating_mul(4) as i64).unsigned_abs()
        + (h as i64 - groups[1].saturating_mul(2) as i64).unsigned_abs();
    Some(d88.min(d84).min(d42))
}

fn is_air_volume_uav_format(fmt: i32) -> bool {
    // R16G16B16A16_FLOAT (10), R11G11B10_FLOAT (26)
    matches!(fmt, 10 | 26)
}

/// Match a 3D volumetric inscattering froxel UAV against dispatch grid dimensions, returning
/// a lower-is-better score when compatible (`None` if incompatible).
fn score_volume_target(rd: &D3D12_RESOURCE_DESC, view_fmt: i32, groups: [u32; 3]) -> Option<u64> {
    if rd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D
        || rd.SampleDesc.Count != 1
        || rd.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
        || !is_air_volume_uav_format(view_fmt)
    {
        return None;
    }
    let w = rd.Width as u32;
    let h = rd.Height;
    let d = rd.DepthOrArraySize as u32;
    if w < 16 || h < 16 || d < 4 {
        return None;
    }
    if groups[0] == 0 || groups[1] <= 1 {
        return Some(u64::MAX - (w as u64).saturating_mul(h as u64).saturating_mul(d as u64));
    }
    let max_w = groups[0].saturating_mul(32);
    let max_h = groups[1].saturating_mul(32);
    let min_w = (groups[0] / 2).max(16);
    let min_h = (groups[1] / 2).max(16);
    if w < min_w || w > max_w || h < min_h || h > max_h {
        return None;
    }
    let dw = (w as i64 - groups[0].saturating_mul(8) as i64)
        .unsigned_abs()
        .min((w as i64 - groups[0].saturating_mul(4) as i64).unsigned_abs());
    let dh = (h as i64 - groups[1].saturating_mul(8) as i64)
        .unsigned_abs()
        .min((h as i64 - groups[1].saturating_mul(4) as i64).unsigned_abs());
    let dd = if groups[2] <= 1 {
        0
    } else {
        (d as i64 - groups[2] as i64)
            .unsigned_abs()
            .min((d as i64 - groups[2].saturating_mul(2) as i64).unsigned_abs())
            .min((d as i64 - groups[2].saturating_mul(4) as i64).unsigned_abs())
            .min((d as i64 - groups[2].saturating_mul(8) as i64).unsigned_abs())
    };
    Some(dw + dh + dd)
}

#[cfg(test)]
fn matches_dispatch_target(rd: &D3D12_RESOURCE_DESC, groups: [u32; 3]) -> bool {
    score_dispatch_target(rd, rd.Format.0, groups).is_some()
}

/// Compute scalar/RGB stand-in sky level (mirrors the GPU `sky_reduce_cs` transfer curve).
#[cfg(test)]
fn sky_transfer(mean_rgb: [f32; 3], gain: f32, power: f32, least: f32) -> [f32; 3] {
    [
        gain * mean_rgb[0].max(0.0).powf(power) + least,
        gain * mean_rgb[1].max(0.0).powf(power) + least,
        gain * mean_rgb[2].max(0.0).powf(power) + least,
    ]
}

// ---------------------------------------------------------------------------------------
// Internal D3D12 compute pipelines & per-device rotating descriptor ring
// ---------------------------------------------------------------------------------------

/// Rotating groups of 3 descriptors (`t0` SRV, `u0` UAV, `u1` UAV) in our private shader-visible
/// heap so back-to-back in-flight dispatches never overwrite descriptors still read by the GPU.
const RING_GROUPS: u32 = 256;
const GROUP_SLOTS: u32 = 3;

struct AmbientDevice {
    rs: ID3D12RootSignature,
    reduce_pso: ID3D12PipelineState,
    fill_pso: ID3D12PipelineState,
    fill_write_pso: ID3D12PipelineState,
    fill_r9_pso: ID3D12PipelineState,
    reflect_pso: ID3D12PipelineState,
    air_pso: ID3D12PipelineState,
    sky_copy: ID3D12Resource,
    sky_level: ID3D12Resource,
    dummy_depth: ID3D12Resource,
    heap: ID3D12DescriptorHeap,
    heap_cpu: u64,
    heap_gpu: u64,
    inc: u32,
    next_group: AtomicU32,
    sky_ready: AtomicBool,
    last_sky_us: AtomicU64,
}

unsafe impl Send for AmbientDevice {}
unsafe impl Sync for AmbientDevice {}

static DEVICES: Mutex<Option<HashMap<usize, Option<std::sync::Arc<AmbientDevice>>>>> = Mutex::new(None);

fn compile_cs_sm5(src: &str) -> Option<Vec<u8>> {
    unsafe {
        let mut blob = None;
        let r = D3DCompile(
            src.as_ptr() as *const c_void,
            src.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(b"cs_5_0\0".as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => Some(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()),
            _ => None,
        }
    }
}

const SKY_REDUCE_HLSL: &str = r#"
Texture2D<float4> sky_tex : register(t0);
RWStructuredBuffer<float4> sky_level : register(u0);
cbuffer C : register(b0) {
    float gain;
    float power;
    float least;
    float settle_alpha;
    uint _u4;
    uint _u5;
    uint _u6;
    uint _u7;
};
groupshared float3 gs_sum[64];
[numthreads(8, 8, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint gidx : SV_GroupIndex) {
    float3 acc = float3(0.0, 0.0, 0.0);
    uint x0 = gtid.x * 16;
    uint y0 = gtid.y * 16;
    [loop]
    for (uint dy = 0; dy < 16; ++dy) {
        [loop]
        for (uint dx = 0; dx < 16; ++dx) {
            float3 c = sky_tex.Load(int3(int(x0 + dx), int(y0 + dy), 0)).rgb;
            if (all(isfinite(c))) {
                acc += clamp(c, 0.0, 16.0);
            }
        }
    }
    gs_sum[gidx] = acc * (1.0 / 256.0);
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 32; s > 0; s >>= 1) {
        if (gidx < s) {
            gs_sum[gidx] += gs_sum[gidx + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (gidx == 0) {
        float3 mean_rgb = max(gs_sum[0] * (1.0 / 64.0), 0.0);
        float3 target = gain * pow(mean_rgb, power) + least;
        float4 prev = sky_level[0];
        float a = saturate(settle_alpha);
        float3 level = (a >= 0.999 || !all(isfinite(prev.rgb)) || prev.w <= 0.0)
            ? target
            : lerp(prev.rgb, target, a);
        float lum = dot(level, float3(0.2126, 0.7152, 0.0722));
        sky_level[0] = float4(level, lum);
    }
}
"#;

/// Screen-space radiance floor + 8-tap IGN-rotated golden-angle horizon crease AO + outlier
/// probe tile suppression (requires typed UAV load on `out_uav`).
const AMBIENT_FILL_HLSL: &str = r#"
Texture2D<float> depth_tex : register(t0);
RWTexture2D<float4> out_uav : register(u0);
RWStructuredBuffer<float4> sky_level : register(u1);
cbuffer C : register(b0) {
    uint width;
    uint height;
    uint depth_w;
    uint depth_h;
    uint has_depth;
    float ao_strength;
    uint sky_ready;
    float fallback_level;
};
float horizon_crease_ao(uint2 tid, int2 dc, int2 maxc) {
    float d0 = depth_tex.Load(int3(dc, 0));
    if (!isfinite(d0) || d0 <= 1e-5 || d0 >= 0.99999) {
        return 1.0;
    }
    float ign = frac(52.9829189 * frac(dot(float2(tid), float2(0.06711056, 0.00583715))));
    float base_ang = ign * 6.2831853;
    float max_rad = clamp(float(depth_w) * 0.012, 4.0, 16.0);
    float inv_d0 = 1.0 / max(d0, 1e-5);
    float occ = 0.0;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        float ang = base_ang + float(i) * 0.78539816;
        float t = frac(ign * 0.61803399 + (float(i) + 0.5) * 0.25);
        float rad = clamp(max_rad * sqrt(t), 1.5, 16.0);
        int2 ofs = int2(int(round(cos(ang) * rad)), int(round(sin(ang) * rad)));
        float dp = depth_tex.Load(int3(clamp(dc + ofs, int2(0, 0), maxc), 0));
        float dn = depth_tex.Load(int3(clamp(dc - ofs, int2(0, 0), maxc), 0));
        float rp = (dp - d0) * inv_d0;
        float rn = (dn - d0) * inv_d0;
        float wp = (isfinite(dp) && dp > 1e-6 && dp < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rp))) : 0.0;
        float wn = (isfinite(dn) && dn > 1e-6 && dn < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rn))) : 0.0;
        float conc = max(0.0, rp * wp + rn * wn);
        occ += saturate((conc - 0.003) * 4.5);
    }
    return saturate(1.0 - ao_strength * saturate(occ * 0.25));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float3 amb = (sky_ready != 0) ? sky_level[0].rgb : float3(fallback_level, fallback_level, fallback_level);
    float ao = 1.0;
    if (has_depth != 0 && depth_w > 0 && depth_h > 0) {
        int2 dc = int2(
            int(min((tid.x * depth_w) / max(width, 1u), depth_w - 1)),
            int(min((tid.y * depth_h) / max(height, 1u), depth_h - 1))
        );
        int2 maxc = int2(int(depth_w) - 1, int(depth_h) - 1);
        ao = horizon_crease_ao(tid.xy, dc, maxc);
    }
    float3 amb_ao = amb * ao;
    float amb_lum = max(dot(amb, float3(0.2126, 0.7152, 0.0722)), 0.02);
    float cap_lum = 4.0 * amb_lum;
    float4 cur = out_uav[tid.xy];
    float3 existing = all(isfinite(cur.rgb)) ? max(cur.rgb, 0.0) : float3(0.0, 0.0, 0.0);
    if (existing.g < 0.5 * min(existing.r, existing.b) && max(existing.r, existing.b) > 0.02) {
        existing = float3(0.0, 0.0, 0.0);
    }
    float cur_lum = dot(existing, float3(0.2126, 0.7152, 0.0722));
    if (cur_lum > cap_lum) {
        existing *= (cap_lum / max(cur_lum, 1e-6));
    }
    float ch_cap = cap_lum * 1.5;
    existing = min(existing, float3(ch_cap, ch_cap, ch_cap));
    float3 out_rgb = max(existing, amb_ao);
    out_uav[tid.xy] = float4(out_rgb, isfinite(cur.a) ? cur.a : 1.0);
}
"#;

/// Screen-space diffuse probe radiance floor + 8-tap IGN-rotated horizon crease AO for
/// `CS_ScreenProbes_TemporalResolve` targets stored as `R9G9B9E5_SHAREDEXP` packed in `R32_UINT`.
const AMBIENT_FILL_R9G9B9E5_HLSL: &str = r#"
Texture2D<float> depth_tex : register(t0);
RWTexture2D<uint> out_uav : register(u0);
RWStructuredBuffer<float4> sky_level : register(u1);
cbuffer C : register(b0) {
    uint width;
    uint height;
    uint depth_w;
    uint depth_h;
    uint has_depth;
    float ao_strength;
    uint sky_ready;
    float fallback_level;
};
float3 unpack_r9g9b9e5(uint v) {
    float scale = exp2(float(v >> 27) - 24.0);
    return float3(float(v & 511u), float((v >> 9) & 511u), float((v >> 18) & 511u)) * scale;
}
uint pack_r9g9b9e5(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float top = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(top, 1e-9)))) + 16, 0, 31);
    float inv_s = exp2(float(24 - e));
    uint3 m = min(uint3(floor(c * inv_s + 0.5)), uint3(511u, 511u, 511u));
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
float horizon_crease_ao(uint2 tid, int2 dc, int2 maxc) {
    float d0 = depth_tex.Load(int3(dc, 0));
    if (!isfinite(d0) || d0 <= 1e-5 || d0 >= 0.99999) {
        return 1.0;
    }
    float ign = frac(52.9829189 * frac(dot(float2(tid), float2(0.06711056, 0.00583715))));
    float base_ang = ign * 6.2831853;
    float max_rad = clamp(float(depth_w) * 0.012, 4.0, 16.0);
    float inv_d0 = 1.0 / max(d0, 1e-5);
    float occ = 0.0;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        float ang = base_ang + float(i) * 0.78539816;
        float t = frac(ign * 0.61803399 + (float(i) + 0.5) * 0.25);
        float rad = clamp(max_rad * sqrt(t), 1.5, 16.0);
        int2 ofs = int2(int(round(cos(ang) * rad)), int(round(sin(ang) * rad)));
        float dp = depth_tex.Load(int3(clamp(dc + ofs, int2(0, 0), maxc), 0));
        float dn = depth_tex.Load(int3(clamp(dc - ofs, int2(0, 0), maxc), 0));
        float rp = (dp - d0) * inv_d0;
        float rn = (dn - d0) * inv_d0;
        float wp = (isfinite(dp) && dp > 1e-6 && dp < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rp))) : 0.0;
        float wn = (isfinite(dn) && dn > 1e-6 && dn < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rn))) : 0.0;
        float conc = max(0.0, rp * wp + rn * wn);
        occ += saturate((conc - 0.003) * 4.5);
    }
    return saturate(1.0 - ao_strength * saturate(occ * 0.25));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float3 raw_amb = (sky_ready != 0) ? sky_level[0].rgb : float3(fallback_level, fallback_level, fallback_level);
    float3 amb = clamp(raw_amb, 0.0, 0.65);
    float ao = 1.0;
    if (has_depth != 0 && depth_w > 0 && depth_h > 0) {
        int2 dc = int2(
            int(min((tid.x * depth_w) / max(width, 1u), depth_w - 1)),
            int(min((tid.y * depth_h) / max(height, 1u), depth_h - 1))
        );
        int2 maxc = int2(int(depth_w) - 1, int(depth_h) - 1);
        ao = horizon_crease_ao(tid.xy, dc, maxc);
    }
    float3 amb_ao = amb * ao;
    float3 existing = unpack_r9g9b9e5(out_uav[tid.xy]);
    if (!all(isfinite(existing))) existing = float3(0.0, 0.0, 0.0);
    if (existing.g < 0.5 * min(existing.r, existing.b) && max(existing.r, existing.b) > 0.02) {
        existing = float3(0.0, 0.0, 0.0);
    }
    float amb_lum = max(dot(amb, float3(0.2126, 0.7152, 0.0722)), 0.02);
    float cap_lum = 6.0 * amb_lum;
    float cur_lum = dot(existing, float3(0.2126, 0.7152, 0.0722));
    if (cur_lum > cap_lum) {
        existing *= (cap_lum / max(cur_lum, 1e-6));
    }
    out_uav[tid.xy] = pack_r9g9b9e5(max(existing, amb_ao));
}
"#;

/// Write-only fallback for `TemporalResolve` when typed UAV load is unsupported on `out_uav`.
const AMBIENT_FILL_WRITE_HLSL: &str = r#"
Texture2D<float> depth_tex : register(t0);
RWTexture2D<float4> out_uav : register(u0);
RWStructuredBuffer<float4> sky_level : register(u1);
cbuffer C : register(b0) {
    uint width;
    uint height;
    uint depth_w;
    uint depth_h;
    uint has_depth;
    float ao_strength;
    uint sky_ready;
    float fallback_level;
};
float horizon_crease_ao(uint2 tid, int2 dc, int2 maxc) {
    float d0 = depth_tex.Load(int3(dc, 0));
    if (!isfinite(d0) || d0 <= 1e-5 || d0 >= 0.99999) {
        return 1.0;
    }
    float ign = frac(52.9829189 * frac(dot(float2(tid), float2(0.06711056, 0.00583715))));
    float base_ang = ign * 6.2831853;
    float max_rad = clamp(float(depth_w) * 0.012, 4.0, 16.0);
    float inv_d0 = 1.0 / max(d0, 1e-5);
    float occ = 0.0;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        float ang = base_ang + float(i) * 0.78539816;
        float t = frac(ign * 0.61803399 + (float(i) + 0.5) * 0.25);
        float rad = clamp(max_rad * sqrt(t), 1.5, 16.0);
        int2 ofs = int2(int(round(cos(ang) * rad)), int(round(sin(ang) * rad)));
        float dp = depth_tex.Load(int3(clamp(dc + ofs, int2(0, 0), maxc), 0));
        float dn = depth_tex.Load(int3(clamp(dc - ofs, int2(0, 0), maxc), 0));
        float rp = (dp - d0) * inv_d0;
        float rn = (dn - d0) * inv_d0;
        float wp = (isfinite(dp) && dp > 1e-6 && dp < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rp))) : 0.0;
        float wn = (isfinite(dn) && dn > 1e-6 && dn < 0.99999) ? (1.0 - smoothstep(0.15, 0.45, abs(rn))) : 0.0;
        float conc = max(0.0, rp * wp + rn * wn);
        occ += saturate((conc - 0.003) * 4.5);
    }
    return saturate(1.0 - ao_strength * saturate(occ * 0.25));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float3 amb = (sky_ready != 0) ? sky_level[0].rgb : float3(fallback_level, fallback_level, fallback_level);
    float ao = 1.0;
    if (has_depth != 0 && depth_w > 0 && depth_h > 0) {
        int2 dc = int2(
            int(min((tid.x * depth_w) / max(width, 1u), depth_w - 1)),
            int(min((tid.y * depth_h) / max(height, 1u), depth_h - 1))
        );
        int2 maxc = int2(int(depth_w) - 1, int(depth_h) - 1);
        ao = horizon_crease_ao(tid.xy, dc, maxc);
    }
    out_uav[tid.xy] = float4(amb * ao, 1.0);
}
"#;

const REFLECT_CLAMP_HLSL: &str = r#"
Texture2D<float> _unused : register(t0);
RWTexture2D<float4> out_uav : register(u0);
RWStructuredBuffer<float4> sky_level : register(u1);
cbuffer C : register(b0) {
    uint width;
    uint height;
    uint _u2;
    uint _u3;
    uint _u4;
    float reflect_scale;
    float _p6;
    float _p7;
};
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float4 cur = out_uav[tid.xy];
    float3 rgb = all(isfinite(cur.rgb)) ? max(cur.rgb, 0.0) : float3(0.0, 0.0, 0.0);
    float3 cap = max(sky_level[0].rgb * reflect_scale, float3(0.005, 0.005, 0.005));
    out_uav[tid.xy] = float4(min(rgb, cap), isfinite(cur.a) ? cur.a : 1.0);
}
"#;

/// Lightweight 3D froxel luminance cap for `CS_AccumulateLocalLightInscatteringVolume`: smoothly
/// compresses over-bright indoor volumetric haze against the settled sky luminance * `air_scale`
/// while preserving sky scatter chromaticity through windows.
const AIR_CAP_HLSL: &str = r#"
Texture2D<float> _unused : register(t0);
RWTexture3D<float4> vol_uav : register(u0);
RWStructuredBuffer<float4> sky_level : register(u1);
cbuffer C : register(b0) {
    uint width;
    uint height;
    uint depth;
    uint _u3;
    uint _u4;
    float air_scale;
    float _p6;
    float _p7;
};
[numthreads(8, 8, 2)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height || tid.z >= depth) return;
    float4 cur = vol_uav[tid];
    float3 rgb = all(isfinite(cur.rgb)) ? max(cur.rgb, 0.0) : float3(0.0, 0.0, 0.0);
    float sky_lum = max(sky_level[0].w, 0.01);
    float cap = max(sky_lum * air_scale, 0.01);
    float lum = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    if (lum > cap) {
        float excess = lum - cap;
        float rolled = cap + excess / (1.0 + excess / max(cap, 1e-4));
        rgb *= (rolled / max(lum, 1e-6));
    }
    float ch_cap = cap * 2.0;
    rgb = min(rgb, float3(ch_cap, ch_cap, ch_cap));
    vol_uav[tid] = float4(rgb, isfinite(cur.a) ? cur.a : 0.0);
}
"#;

unsafe fn build_device(dev: &ID3D12Device) -> Option<AmbientDevice> {
    let reduce_bc = compile_cs_sm5(SKY_REDUCE_HLSL)?;
    let fill_bc = compile_cs_sm5(AMBIENT_FILL_HLSL)?;
    let fill_write_bc = compile_cs_sm5(AMBIENT_FILL_WRITE_HLSL)?;
    let fill_r9_bc = compile_cs_sm5(AMBIENT_FILL_R9G9B9E5_HLSL)?;
    let reflect_bc = compile_cs_sm5(REFLECT_CLAMP_HLSL)?;
    let air_bc = compile_cs_sm5(AIR_CAP_HLSL)?;

    // Shared compute RS: [0] 8 root constants (b0), [1] 1 SRV (t0), [2] 2 UAVs (u0..u1)
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 2,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 8 },
            },
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &srv_range },
            },
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &uav_range },
            },
        },
    ];
    let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut rs_blob = None;
    D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None).ok()?;
    let rs_blob = rs_blob?;

    let dev_raw = dev.as_raw();
    type CreateRsFn = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const c_void, *mut *mut c_void) -> i32;
    type CreateCsFn = unsafe extern "system" fn(*mut c_void, *const D3D12_COMPUTE_PIPELINE_STATE_DESC, *const c_void, *mut *mut c_void) -> i32;
    type CreateHeapFn = unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const c_void, *mut *mut c_void) -> i32;
    type IncSizeFn = unsafe extern "system" fn(*mut c_void, D3D12_DESCRIPTOR_HEAP_TYPE) -> u32;
    type CreateResFn = unsafe extern "system" fn(
        *mut c_void,
        *const D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_FLAGS,
        *const D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_STATES,
        *const D3D12_CLEAR_VALUE,
        *const c_void,
        *mut *mut c_void,
    ) -> i32;

    let create_rs: CreateRsFn = std::mem::transmute(real::orig_fn(dev_raw, 16));
    let create_cs: CreateCsFn = std::mem::transmute(real::orig_fn(dev_raw, 11));
    let create_heap: CreateHeapFn = std::mem::transmute(real::orig_fn(dev_raw, 14));
    let inc_size: IncSizeFn = std::mem::transmute(real::orig_fn(dev_raw, 15));
    let create_res: CreateResFn = std::mem::transmute(real::orig_fn(dev_raw, 27));

    let mut rs_raw = std::ptr::null_mut();
    if create_rs(dev_raw, 0, rs_blob.GetBufferPointer(), rs_blob.GetBufferSize(), &ID3D12RootSignature::IID as *const _ as *const c_void, &mut rs_raw) < 0 || rs_raw.is_null() {
        return None;
    }
    let rs = ID3D12RootSignature::from_raw(rs_raw);

    let make_cs = |bc: &[u8], name: &str| -> Option<ID3D12PipelineState> {
        let mut d = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(rs.clone())),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bc.as_ptr() as *const c_void, BytecodeLength: bc.len() },
            ..Default::default()
        };
        let mut raw = std::ptr::null_mut();
        let hr = create_cs(dev_raw, &d, &ID3D12PipelineState::IID as *const _ as *const c_void, &mut raw);
        ManuallyDrop::drop(&mut d.pRootSignature);
        (hr >= 0 && !raw.is_null()).then(|| {
            let pso = ID3D12PipelineState::from_raw(raw);
            let _ = pso.SetName(&windows::core::HSTRING::from(name));
            super::dred::label_text(pso.as_raw() as usize, || name.to_string());
            pso
        })
    };
    let reduce_pso = make_cs(&reduce_bc, "xcl ambient sky_reduce")?;
    let fill_pso = make_cs(&fill_bc, "xcl ambient fill")?;
    let fill_write_pso = make_cs(&fill_write_bc, "xcl ambient fill_write")?;
    let fill_r9_pso = make_cs(&fill_r9_bc, "xcl ambient fill_r9")?;
    let reflect_pso = make_cs(&reflect_bc, "xcl ambient reflect_clamp")?;
    let air_pso = make_cs(&air_bc, "xcl ambient air_cap")?;

    let default_hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let make_res = |rd: &D3D12_RESOURCE_DESC, init: D3D12_RESOURCE_STATES| -> Option<ID3D12Resource> {
        let mut raw = std::ptr::null_mut();
        let hr = create_res(dev_raw, &default_hp, D3D12_HEAP_FLAG_NONE, rd, init, std::ptr::null(), &ID3D12Resource::IID as *const _ as *const c_void, &mut raw);
        (hr >= 0 && !raw.is_null()).then(|| ID3D12Resource::from_raw(raw))
    };

    let sky_copy = make_res(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 128,
            Height: 128,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    )?;
    let sky_level = make_res(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 256,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let dummy_depth = make_res(
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
    )?;

    let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        NumDescriptors: RING_GROUPS * GROUP_SLOTS,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
        NodeMask: 0,
    };
    let mut heap_raw = std::ptr::null_mut();
    if create_heap(dev_raw, &heap_desc, &ID3D12DescriptorHeap::IID as *const _ as *const c_void, &mut heap_raw) < 0 || heap_raw.is_null() {
        return None;
    }
    let heap = ID3D12DescriptorHeap::from_raw(heap_raw);
    let heap_cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr as u64;
    let heap_gpu = heap.GetGPUDescriptorHandleForHeapStart().ptr;
    let inc = inc_size(dev_raw, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    Some(AmbientDevice {
        rs,
        reduce_pso,
        fill_pso,
        fill_write_pso,
        fill_r9_pso,
        reflect_pso,
        air_pso,
        sky_copy,
        sky_level,
        dummy_depth,
        heap,
        heap_cpu,
        heap_gpu,
        inc,
        next_group: AtomicU32::new(0),
        sky_ready: AtomicBool::new(false),
        last_sky_us: AtomicU64::new(0),
    })
}

unsafe fn get_device(dev: &ID3D12Device) -> Option<std::sync::Arc<AmbientDevice>> {
    let key = dev.as_raw() as usize;
    {
        let g = DEVICES.lock().unwrap();
        if let Some(entry) = g.as_ref().and_then(|m| m.get(&key)) {
            return entry.clone();
        }
    }
    let built = build_device(dev).map(std::sync::Arc::new);
    DEVICES.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, built.clone());
    built
}

impl AmbientDevice {
    fn alloc_group(&self) -> (u64, u64) {
        let g = self.next_group.fetch_add(1, Ordering::Relaxed) % RING_GROUPS;
        let off = (g * GROUP_SLOTS) as u64 * self.inc as u64;
        (self.heap_cpu + off, self.heap_gpu + off)
    }
}

unsafe fn uav_barrier(list: *mut c_void, res_raw: usize) {
    let b = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::transmute_copy(&res_raw),
            }),
        },
    };
    list_x::pc_resource_barriers(list, &[b]);
}

unsafe fn typed_uav_load_supported(dev: &ID3D12Device, fmt: DXGI_FORMAT) -> bool {
    let mut fs = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: fmt, ..Default::default() };
    dev.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mut fs as *mut _ as *mut c_void, std::mem::size_of_val(&fs) as u32).is_ok()
        && (fs.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD.0 != 0)
}

/// Called right after a translated graphics draw completes: if the PSO is the sky paraboloid
/// reflection shader targeting a `128x128 R16G16B16A16_FLOAT` 2D render target, snapshot it
/// and update the sky-following ambient level buffer on the GPU.
pub(super) unsafe fn after_draw(list: *mut c_void, pso_resolved: usize) {
    if !has_sky_psos() || !active() || role_of(pso_resolved) != Some(PassRole::SkyDome) {
        return;
    }
    if list_x::tracked_list_type(list) != Some(D3D12_COMMAND_LIST_TYPE_DIRECT) {
        return;
    }
    let Some(rt0) = present::rt0_resource(list as usize) else { return };
    let rt_ptr = rt0 as *mut c_void;
    let Some(rt_res) = ID3D12Resource::from_raw_borrowed(&rt_ptr) else { return };
    if !is_sky_dome_target(&rt_res.GetDesc()) {
        return;
    }
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let mut dev: Option<ID3D12Device> = None;
    if cl.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let Some(ad) = get_device(&dev) else { return };

    let sky_raw = ad.sky_copy.as_raw() as usize;
    list_x::transition(list, rt0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_x::transition(list, sky_raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    list_x::pc_copy_resource(list, sky_raw, rt0);
    list_x::transition(list, sky_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    list_x::transition(list, rt0, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);

    let (cpu0, gpu0) = ad.alloc_group();
    let inc = ad.inc as u64;
    let dev_raw = dev.as_raw();
    type CreateSrvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, usize);
    type CreateUavFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, usize);
    let create_srv: CreateSrvFn = std::mem::transmute(real::orig_fn(dev_raw, 18));
    let create_uav: CreateUavFn = std::mem::transmute(real::orig_fn(dev_raw, 19));

    let sky_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
        },
    };
    let level_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 1, StructureByteStride: 16, ..Default::default() },
        },
    };
    create_srv(dev_raw, ad.sky_copy.as_raw(), &sky_srv, cpu0 as usize);
    create_uav(dev_raw, ad.sky_level.as_raw(), std::ptr::null_mut(), &level_uav, (cpu0 + inc) as usize);
    create_uav(dev_raw, ad.sky_level.as_raw(), std::ptr::null_mut(), &level_uav, (cpu0 + 2 * inc) as usize);

    let heap_raw = ad.heap.as_raw() as usize;
    let set_heaps: unsafe extern "system" fn(*mut c_void, u32, *const usize) = std::mem::transmute(real::orig_fn(list, 28));
    let set_cs_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 29));
    let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 25));
    let set_cs_table: unsafe extern "system" fn(*mut c_void, u32, u64) = std::mem::transmute(real::orig_fn(list, 31));
    let set_cs_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) = std::mem::transmute(real::orig_fn(list, 35));
    let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) = std::mem::transmute(real::orig_fn(list, 14));

    let cfg = config();
    let now_us = monotonic_us();
    let prev_us = ad.last_sky_us.swap(now_us, Ordering::AcqRel);
    let first_sample = prev_us == 0 || !ad.sky_ready.load(Ordering::Acquire);
    let dt_s = if first_sample { 0.0 } else { now_us.saturating_sub(prev_us) as f32 * 1e-6 };
    let alpha = settle_alpha(dt_s, cfg.settle_ms, first_sample);
    if !first_sample {
        uav_barrier(list, ad.sky_level.as_raw() as usize);
    }
    let consts = [cfg.gain.to_bits(), cfg.power.to_bits(), cfg.least.to_bits(), alpha.to_bits(), 0, 0, 0, 0];
    set_heaps(list, 1, &heap_raw);
    set_cs_rs(list, ad.rs.as_raw());
    super::dred::mark_pipeline(list, ad.reduce_pso.as_raw() as usize, ad.reduce_pso.as_raw() as usize);
    set_pso(list, ad.reduce_pso.as_raw());
    set_cs_consts(list, 0, 8, consts.as_ptr(), 0);
    set_cs_table(list, 1, gpu0);
    set_cs_table(list, 2, gpu0 + inc);
    dispatch(list, 1, 1, 1);
    uav_barrier(list, ad.sky_level.as_raw() as usize);
    ad.sky_ready.store(true, Ordering::Release);
    list_x::restore_after_internal_pass(list, false, true);

    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) == 0 {
        crate::xlog_info!("D3D12.X: ambient: captured 128x128 R16G16B16A16_FLOAT sky dome ({rt0:#x})");
    }
}

/// Maximum bindless descriptor-heap slot index probed from root constants/CBVs
/// (covers >1.24M-slot console shader-visible view heaps).
const MAX_SLOT_INDEX: u32 = 2_000_000;

/// Decode a CBV / buffer `V#` descriptor (`gpu::desc::cbv_words` / `buffer_words`) into
/// `(va, byte_len)` if it is a readable buffer descriptor (`w3 >> 28 < 8` and dst_sel `0xFAC`).
fn decode_cbv_vsharp(w: [u32; 4]) -> Option<(usize, usize)> {
    if (w[3] >> 28) >= 8 || (w[3] & 0xFFF) != 0xFAC {
        return None;
    }
    let va = (w[0] as u64 | ((w[1] as u64 & 0xFFFF) << 32)) as usize;
    if va == 0 {
        return None;
    }
    let stride = ((w[1] >> 16) & 0x3FFF) as usize;
    let records = w[2] as usize;
    let bytes = if stride == 0 { records } else { records.saturating_mul(stride) };
    (bytes >= 4).then_some((va, bytes.min(2048)))
}

/// Resolve a GPU VA to a CPU-readable host address when it lives in a `PAGE_NOACCESS` GPU mirror view.
fn resolve_cbv_va(va: usize) -> usize {
    if va == 0 {
        return 0;
    }
    if crate::gpu::shader::readable(va, 4) {
        return va;
    }
    match crate::hle::kernelx::mirrors::mapped_source(va) {
        Some(Some(backing)) => backing,
        Some(None) => 0,
        None => crate::hle::kernelx::mirrors::source(va).unwrap_or(va),
    }
}

/// Scan up to `max_bytes` (capped at 2048 B) of readable constant-buffer memory at `va`,
/// resolving each chunk (capped at 256 B and never crossing a 64 KiB mirror page boundary)
/// through `resolve`, and feed candidate bindless descriptor indices `< MAX_SLOT_INDEX` into `push_idx`.
unsafe fn scan_cbv_indices_with(
    va: usize,
    max_bytes: usize,
    mut resolve: impl FnMut(usize) -> usize,
    push_idx: &mut impl FnMut(u32),
) {
    if va == 0 || max_bytes < 4 {
        return;
    }
    const PAGE: usize = crate::hle::kernelx::mirrors::PAGE;
    let total = max_bytes.min(2048);
    let mut off = 0usize;
    while off < total {
        let Some(cur_va) = va.checked_add(off) else { break };
        let to_page_end = PAGE - (cur_va & (PAGE - 1));
        let chunk = (total - off).min(256).min(to_page_end);
        let base = resolve(cur_va);
        if base == 0 || !crate::gpu::shader::readable(base, chunk) {
            break;
        }
        let words = chunk / 4;
        for k in 0..words {
            let v = ((base + 4 * k) as *const u32).read_unaligned();
            if v != 0 {
                push_idx(v);
            }
        }
        off += chunk;
    }
}

/// Scan up to `max_bytes` (capped at 2048 B) of readable constant-buffer memory at `va` in
/// page-bounded chunks and feed candidate bindless descriptor indices `< MAX_SLOT_INDEX` into `push_idx`.
unsafe fn scan_cbv_indices(va: usize, max_bytes: usize, push_idx: &mut impl FnMut(u32)) {
    scan_cbv_indices_with(va, max_bytes, resolve_cbv_va, push_idx);
}

/// Inspect a compute dispatch's bound descriptor tables, SM 6.6 directly-indexed view heap
/// (`view_heap_base`), and root/table CBVs to locate its output float UAV (2D screen-space or
/// 3D froxel volume when `want_3d` is true) and optional depth SRV for crease AO.
unsafe fn find_dispatch_views(
    role: PassRole,
    layout: &Layout,
    args: &[u8],
    groups: [u32; 3],
    view_heap_base: u32,
    want_3d: bool,
) -> (Option<(usize, i32, u32, u32, u32)>, Option<(usize, i32, u32, u32)>, usize, usize, usize) {
    let Some(u) = crate::gpu::uma::get() else { return (None, None, 0, 0, 0) };
    let word = |at: usize| args.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let mut param_tables: Vec<u32> = Vec::new();
    let mut tables: Vec<u32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut push_idx = |v: u32| {
        if v < MAX_SLOT_INDEX && seen.insert(v) {
            indices.push(v);
        }
    };
    if view_heap_base != 0 {
        tables.push(view_heap_base);
    }
    for p in &layout.params {
        match *p {
            Param::Constants { at, count } => {
                for k in 0..count.min(64) as usize {
                    if let Some(v) = word(at as usize + 4 * k) {
                        push_idx(v);
                    }
                }
            }
            Param::Descriptor { at } => {
                let lo = word(at as usize).unwrap_or(0) as u64;
                let hi = word(at as usize + 4).unwrap_or(0) as u64;
                let va = (lo | (hi << 32)) as usize;
                scan_cbv_indices(va, 2048, &mut push_idx);
            }
            Param::Table { at } => {
                if let Some(t) = word(at as usize) {
                    if t != 0 {
                        if !param_tables.contains(&t) {
                            param_tables.push(t);
                        }
                        if !tables.contains(&t) {
                            tables.push(t);
                        }
                    }
                }
            }
        }
    }

    // Scan direct slots `0..64` of every bound `Param::Table` for BOTH direct image T# descriptors
    // and CBV V# descriptors (whose buffer contents often carry bindless indices into `view_heap`).
    let mut table_cbvs = 0usize;
    let mut direct_addrs: Vec<usize> = Vec::new();
    for &t in &param_tables {
        for k in 0..64usize {
            let Some(off) = (t as usize).checked_add(32 * k) else { break };
            let Some(addr) = u.base().checked_add(off) else { break };
            if !u.contains(addr) || !crate::gpu::shader::readable(addr, 16) {
                break;
            }
            let w = [
                (addr as *const u32).read_unaligned(),
                ((addr + 4) as *const u32).read_unaligned(),
                ((addr + 8) as *const u32).read_unaligned(),
                ((addr + 12) as *const u32).read_unaligned(),
            ];
            if (w[3] >> 28) >= 8 {
                direct_addrs.push(addr);
            } else if let Some((cbv_va, cbv_len)) = decode_cbv_vsharp(w) {
                table_cbvs += 1;
                scan_cbv_indices(cbv_va, cbv_len, &mut push_idx);
            }
        }
    }

    let mut uav_hit: Option<(u64, usize, i32, u32, u32, u32)> = None;
    let mut depth_hit: Option<(usize, i32, u32, u32)> = None;
    let slots = VIEW_SLOTS.read().unwrap();
    let Some(vs) = slots.as_ref() else { return (None, None, tables.len(), indices.len(), table_cbvs) };

    let mut check_slot = |addr: usize| {
        if !u.contains(addr) || !crate::gpu::shader::readable(addr, 16) {
            return;
        }
        let w0 = (addr as *const u32).read_unaligned();
        let w3 = ((addr + 12) as *const u32).read_unaligned();
        // Image T# descriptors have type >= 8 in bits [31:28] of word 3.
        if (w3 >> 28) < 8 {
            return;
        }
        let Some(&ve) = vs
            .by_addr
            .get(&addr)
            .or_else(|| (w0 != 0).then(|| vs.by_idx.get(&w0)).flatten())
        else {
            return;
        };
        let ptr = ve.res as *mut c_void;
        let Some(res) = ID3D12Resource::from_raw_borrowed(&ptr) else { return };
        let rd = res.GetDesc();
        if ve.is_uav {
            let scored = if want_3d {
                score_volume_target(&rd, ve.fmt, groups).map(|s| (s, rd.DepthOrArraySize as u32))
            } else {
                score_dispatch_target_for_role(role, &rd, ve.fmt, groups).map(|s| (s, 1u32))
            };
            if let Some((score, depth)) = scored {
                if uav_hit.is_none_or(|(best, _, _, _, _, _)| score < best) {
                    uav_hit = Some((score, ve.res, ve.fmt, rd.Width as u32, rd.Height, depth));
                }
            }
        } else if !want_3d
            && depth_hit.is_none()
            && rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
            && rd.DepthOrArraySize == 1
            && rd.SampleDesc.Count == 1
            && rd.Width >= 64
            && rd.Height >= 64
        {
            depth_hit = Some((ve.res, ve.fmt, rd.Width as u32, rd.Height));
        }
    };

    for addr in direct_addrs {
        check_slot(addr);
    }
    for &t in &tables {
        for &idx in &indices {
            let Some(off) = (t as usize).checked_add(32usize.saturating_mul(idx as usize)) else { continue };
            let Some(addr) = u.base().checked_add(off) else { continue };
            check_slot(addr);
        }
    }
    (uav_hit.map(|(_, r, f, w, h, d)| (r, f, w, h, d)), depth_hit, tables.len(), indices.len(), table_cbvs)
}

/// Called right after a translated compute dispatch completes: when RT is gated off and a sky
/// dome sample has been captured, apply the sky-following screen-space radiance floor + 8-tap
/// horizon crease AO on `CS_ScreenProbes_TemporalResolve`, clamp `CS_UpdateHistory` reflections
/// to `reflect * ambient`, and cap `CS_AccumulateLocalLightInscatteringVolume` 3D froxel haze
/// against settled sky luminance * `air`.
pub(super) unsafe fn after_compute_dispatch(
    list: *mut c_void,
    pso_resolved: usize,
    groups: [u32; 3],
    layout: Option<&Layout>,
    args: &[u8],
    view_heap_base: u32,
) {
    if !has_compute_psos() || !active() {
        return;
    }
    let Some(role) = role_of(pso_resolved) else { return };
    if !matches!(role, PassRole::TemporalResolve | PassRole::ReflectionHistory | PassRole::InscatteringVolume) {
        return;
    }
    let cfg = config();
    if role == PassRole::InscatteringVolume && cfg.air <= 0.0 {
        return;
    }
    let ty = list_x::tracked_list_type(list);
    if !matches!(ty, Some(D3D12_COMMAND_LIST_TYPE_DIRECT | D3D12_COMMAND_LIST_TYPE_COMPUTE)) {
        static BAD_TY: AtomicU64 = AtomicU64::new(0);
        if BAD_TY.fetch_add(1, Ordering::Relaxed) < 2 {
            crate::xlog_info!("D3D12.X: ambient: {role:?} skipped on unsupported list type {ty:?}");
        }
        return;
    }
    let Some(layout) = layout else {
        static NO_LAYOUT: AtomicU64 = AtomicU64::new(0);
        if NO_LAYOUT.fetch_add(1, Ordering::Relaxed) < 2 {
            crate::xlog_info!("D3D12.X: ambient: {role:?} skipped (no compute root-signature layout)");
        }
        return;
    };
    let Some(cl) = ID3D12GraphicsCommandList::from_raw_borrowed(&list) else { return };
    let mut dev: Option<ID3D12Device> = None;
    if cl.GetDevice(&mut dev).is_err() {
        return;
    }
    let Some(dev) = dev else { return };
    let Some(ad) = get_device(&dev) else { return };
    let sky_ready = ad.sky_ready.load(Ordering::Acquire);
    if !sky_ready {
        static NO_SKY: AtomicU64 = AtomicU64::new(0);
        if NO_SKY.fetch_add(1, Ordering::Relaxed) < 2 {
            crate::xlog_info!(
                "D3D12.X: ambient: {role:?} dispatched ({groups:?}) before sky dome capture ({})",
                if role == PassRole::TemporalResolve { "using least fallback" } else { "skipping" }
            );
        }
        if role != PassRole::TemporalResolve {
            return;
        }
    }

    let want_3d = role == PassRole::InscatteringVolume;
    let (uav_opt, depth_opt, n_tables, n_indices, n_cbvs) = find_dispatch_views(role, layout, args, groups, view_heap_base, want_3d);
    let Some((uav_res, uav_fmt, w, h, d)) = uav_opt else {
        static MISS: AtomicU64 = AtomicU64::new(0);
        if MISS.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_info!(
                "D3D12.X: ambient: {role:?} ({groups:?}) found no matching UAV (tables={n_tables}, view_heap={view_heap_base:#x}, table_cbvs={n_cbvs}, indices={n_indices})"
            );
        }
        return;
    };
    let can_typed_load = typed_uav_load_supported(&dev, DXGI_FORMAT(uav_fmt));
    if matches!(role, PassRole::ReflectionHistory | PassRole::InscatteringVolume) && !can_typed_load {
        static NO_LOAD: AtomicU64 = AtomicU64::new(0);
        if NO_LOAD.fetch_add(1, Ordering::Relaxed) < 2 {
            crate::xlog_info!("D3D12.X: ambient: {role:?} skipped (typed UAV load unsupported for fmt {uav_fmt})");
        }
        return;
    }

    let (cpu0, gpu0) = ad.alloc_group();
    let inc = ad.inc as u64;
    let dev_raw = dev.as_raw();
    type CreateSrvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, usize);
    type CreateUavFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, usize);
    let create_srv: CreateSrvFn = std::mem::transmute(real::orig_fn(dev_raw, 18));
    let create_uav: CreateUavFn = std::mem::transmute(real::orig_fn(dev_raw, 19));

    let (depth_raw, depth_fmt, depth_w, depth_h, has_depth) = match (role, depth_opt) {
        (PassRole::TemporalResolve, Some((r, f, dw, dh))) => (r as *mut c_void, DXGI_FORMAT(f), dw, dh, 1u32),
        _ => (ad.dummy_depth.as_raw(), DXGI_FORMAT_R32_FLOAT, 1, 1, 0u32),
    };
    let depth_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: depth_fmt,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
        },
    };
    let out_uav_desc = if want_3d {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT(uav_fmt),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: d },
            },
        }
    } else {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT(uav_fmt),
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 },
            },
        }
    };
    let level_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_UNKNOWN,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 1, StructureByteStride: 16, ..Default::default() },
        },
    };
    create_srv(dev_raw, depth_raw, &depth_srv, cpu0 as usize);
    create_uav(dev_raw, uav_res as *mut c_void, std::ptr::null_mut(), &out_uav_desc, (cpu0 + inc) as usize);
    create_uav(dev_raw, ad.sky_level.as_raw(), std::ptr::null_mut(), &level_uav, (cpu0 + 2 * inc) as usize);

    uav_barrier(list, uav_res);

    let heap_raw = ad.heap.as_raw() as usize;
    let set_heaps: unsafe extern "system" fn(*mut c_void, u32, *const usize) = std::mem::transmute(real::orig_fn(list, 28));
    let set_cs_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 29));
    let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list, 25));
    let set_cs_table: unsafe extern "system" fn(*mut c_void, u32, u64) = std::mem::transmute(real::orig_fn(list, 31));
    let set_cs_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) = std::mem::transmute(real::orig_fn(list, 35));
    let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) = std::mem::transmute(real::orig_fn(list, 14));

    let (pso, consts, disp) = match role {
        PassRole::TemporalResolve => (
            if uav_fmt == 42 {
                ad.fill_r9_pso.as_raw()
            } else if can_typed_load {
                ad.fill_pso.as_raw()
            } else {
                ad.fill_write_pso.as_raw()
            },
            [w, h, depth_w, depth_h, has_depth, cfg.ao.to_bits(), sky_ready as u32, cfg.least.to_bits()],
            (w.div_ceil(8), h.div_ceil(8), 1),
        ),
        PassRole::ReflectionHistory => (
            ad.reflect_pso.as_raw(),
            [w, h, 0, 0, 0, cfg.reflect.to_bits(), 0, 0],
            (w.div_ceil(8), h.div_ceil(8), 1),
        ),
        PassRole::InscatteringVolume => (
            ad.air_pso.as_raw(),
            [w, h, d, 0, 0, cfg.air.to_bits(), 0, 0],
            (w.div_ceil(8), h.div_ceil(8), d.div_ceil(2)),
        ),
        PassRole::SkyDome => return,
    };
    set_heaps(list, 1, &heap_raw);
    set_cs_rs(list, ad.rs.as_raw());
    super::dred::mark_pipeline(list, pso as usize, pso as usize);
    set_pso(list, pso);
    set_cs_consts(list, 0, 8, consts.as_ptr(), 0);
    set_cs_table(list, 1, gpu0);
    set_cs_table(list, 2, gpu0 + inc);
    dispatch(list, disp.0, disp.1, disp.2);
    uav_barrier(list, uav_res);
    list_x::restore_after_internal_pass(list, false, true);

    static LOGGED: AtomicU64 = AtomicU64::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) < 6 {
        crate::xlog_info!("D3D12.X: ambient: applied {role:?} stand-in on {w}x{h}x{d} fmt {uav_fmt} (depth AO: {has_depth})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stand-in is for a picture without bounce light: with traversal held back, and as much
    /// with traversal running while every ray misses; not with rays that hit.
    #[test]
    fn the_stand_in_is_on_whenever_the_rays_find_nothing() {
        assert!(rays_find_nothing(false, false), "traversal held back");
        assert!(rays_find_nothing(true, true), "every ray misses");
        assert!(rays_find_nothing(false, true));
        assert!(!rays_find_nothing(true, false), "rays that hit bring the title's own light");
    }

    #[test]
    fn mode_and_shader_classification_match_expected_passes() {
        assert_eq!(parse_mode(None), Mode::Auto);
        assert_eq!(parse_mode(Some("auto")), Mode::Auto);
        assert_eq!(parse_mode(Some("sky")), Mode::Sky);
        assert_eq!(parse_mode(Some("1")), Mode::Sky);
        assert_eq!(parse_mode(Some("0")), Mode::Off);
        assert_eq!(parse_mode(Some("off")), Mode::Off);

        assert_eq!(classify_ps(b"\0PRIVps_sky_paraboloid_reflection\0"), Some(PassRole::SkyDome));
        assert_eq!(classify_ps(b"\0PS_Sky_Paraboloid_Reflection_V2\0"), Some(PassRole::SkyDome));
        assert_eq!(classify_ps(b"\0ps_gbuffer_opaque\0"), None);

        assert_eq!(classify_cs(b"ILDN\0CS_ScreenProbes_TemporalResolve\0"), Some(PassRole::TemporalResolve));
        assert_eq!(classify_cs(b"ILDN\0CS_UpdateHistory_16\0"), Some(PassRole::ReflectionHistory));
        assert_eq!(classify_cs(b"ILDN\0CS_CullInstances\0"), None);
    }

    #[test]
    fn sky_dome_filter_rejects_r11g11b10_and_wrong_sizes() {
        let mut rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 128,
            Height: 128,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        assert!(is_sky_dome_target(&rd));

        // Typeless 128x128 RGBA16 render targets are copy-compatible with R16G16B16A16_FLOAT.
        rd.Format = DXGI_FORMAT_R16G16B16A16_TYPELESS;
        assert!(is_sky_dome_target(&rd));

        // Must ignore the brighter 128x128 R11G11B10_FLOAT pass to prevent frame flicker.
        rd.Format = DXGI_FORMAT_R11G11B10_FLOAT;
        assert!(!is_sky_dome_target(&rd));

        rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        rd.Width = 256;
        assert!(!is_sky_dome_target(&rd));

        assert!(is_ambient_compute(b"ILDN\0CS_ScreenProbes_TemporalResolve\0"));
        assert!(is_ambient_compute(b"ILDN\0CS_UpdateHistory_16\0"));
        assert!(!is_ambient_compute(b"ILDN\0CS_CullInstances\0"));
    }

    #[test]
    fn dispatch_target_matching_and_sky_transfer_curve() {
        let rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1280,
            Height: 720,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        assert!(matches_dispatch_target(&rd, [160, 90, 1]));
        assert!(!matches_dispatch_target(&rd, [16, 16, 1]));

        let night = sky_transfer([0.0, 0.0, 0.0], 0.55, 0.65, 0.025);
        assert!((night[0] - 0.025).abs() < 1e-6);
        let day = sky_transfer([1.0, 0.9, 1.2], 0.55, 0.65, 0.025);
        assert!(day[2] > day[0] && day[0] > night[0]);
    }

    #[test]
    fn sm5_ambient_hlsl_shaders_compile_cleanly() {
        assert!(compile_cs_sm5(SKY_REDUCE_HLSL).is_some(), "SKY_REDUCE_HLSL failed D3DCompile");
        assert!(compile_cs_sm5(AMBIENT_FILL_HLSL).is_some(), "AMBIENT_FILL_HLSL failed D3DCompile");
        assert!(compile_cs_sm5(AMBIENT_FILL_WRITE_HLSL).is_some(), "AMBIENT_FILL_WRITE_HLSL failed D3DCompile");
        assert!(compile_cs_sm5(AMBIENT_FILL_R9G9B9E5_HLSL).is_some(), "AMBIENT_FILL_R9G9B9E5_HLSL failed D3DCompile");
        assert!(compile_cs_sm5(REFLECT_CLAMP_HLSL).is_some(), "REFLECT_CLAMP_HLSL failed D3DCompile");
        assert!(compile_cs_sm5(AIR_CAP_HLSL).is_some(), "AIR_CAP_HLSL failed D3DCompile");
    }

    #[test]
    fn temporal_resolve_targets_r32_uint_and_never_hdr_scene_color() {
        let hdr_scene = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 2560,
            Height: 1440,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R11G11B10_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let r9_probe = D3D12_RESOURCE_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ..hdr_scene
        };
        assert!(score_dispatch_target_for_role(PassRole::TemporalResolve, &hdr_scene, DXGI_FORMAT_R11G11B10_FLOAT.0, [320, 360, 1]).is_none());
        assert!(score_dispatch_target_for_role(PassRole::TemporalResolve, &hdr_scene, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [320, 360, 1]).is_none());
        assert!(score_dispatch_target_for_role(PassRole::TemporalResolve, &r9_probe, DXGI_FORMAT_R32_UINT.0, [320, 360, 1]).is_some());
        assert!(score_dispatch_target_for_role(PassRole::ReflectionHistory, &r9_probe, DXGI_FORMAT_R32_UINT.0, [320, 360, 1]).is_none());
        assert!(score_dispatch_target_for_role(PassRole::ReflectionHistory, &hdr_scene, DXGI_FORMAT_R11G11B10_FLOAT.0, [320, 360, 1]).is_some());
    }

    #[test]
    fn inscattering_volume_classification_volume_scoring_and_time_based_settling() {
        assert_eq!(
            classify_cs(b"ILDN\0CS_AccumulateLocalLightInscatteringVolume\0"),
            Some(PassRole::InscatteringVolume)
        );
        assert!(is_ambient_compute(b"ILDN\0CS_AccumulateLocalLightInscatteringVolume_V2\0"));

        let def = Config::default();
        assert!((def.settle_ms - 250.0).abs() < 1e-6);
        assert!((def.air - 0.0).abs() < 1e-6);

        // First capture or settle_ms == 0 snaps immediately; subsequent samples follow 1 - exp(-dt/tau).
        assert_eq!(settle_alpha(0.016, 250.0, true), 1.0);
        assert_eq!(settle_alpha(0.016, 0.0, false), 1.0);
        assert_eq!(settle_alpha(0.0, 250.0, false), 0.0);
        let a_frame = settle_alpha(1.0 / 60.0, 250.0, false);
        let a_tau = settle_alpha(0.25, 250.0, false);
        let a_long = settle_alpha(1.0, 250.0, false);
        assert!(a_frame > 0.05 && a_frame < 0.08, "16.7 ms EMA step: {a_frame}");
        assert!((a_tau - (1.0 - (-1.0f32).exp())).abs() < 1e-5, "dt == tau: {a_tau}");
        assert!(a_long > a_tau && a_long < 1.0, "1 s EMA step: {a_long}");

        // 3D froxel volume matching: accepts 3D RGBA16F and R11G11B10F UAVs, rejects 2D and wrong formats.
        let vol = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
            Width: 160,
            Height: 90,
            DepthOrArraySize: 64,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        assert!(score_volume_target(&vol, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [20, 12, 32]).is_some());
        assert!(score_volume_target(&vol, DXGI_FORMAT_R11G11B10_FLOAT.0, [0, 0, 0]).is_some());
        assert!(score_volume_target(&vol, DXGI_FORMAT_R32G32B32A32_FLOAT.0, [20, 12, 32]).is_none());
        let tex2d = D3D12_RESOURCE_DESC { Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D, ..vol };
        assert!(score_volume_target(&tex2d, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [20, 12, 1]).is_none());
        let tiny = D3D12_RESOURCE_DESC { DepthOrArraySize: 2, ..vol };
        assert!(score_volume_target(&tiny, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [20, 12, 1]).is_none());
    }

    #[test]
    fn typeless_uav_indirect_dispatch_and_cpu_heap_slot_copy_resolve_cleanly() {
        // Typeless resource with concrete R16G16B16A16_FLOAT UAV view must match (both direct and indirect).
        let typeless = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1920,
            Height: 1080,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        assert!(score_dispatch_target(&typeless, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [240, 135, 1]).is_some());
        assert!(score_dispatch_target(&typeless, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [0, 0, 0]).is_some());
        // Texture2DArray must be rejected for TEXTURE2D view creation.
        let arr = D3D12_RESOURCE_DESC { DepthOrArraySize: 4, ..typeless };
        assert!(score_dispatch_target(&arr, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [240, 135, 1]).is_none());

        // [320, 360, 1] Wave32 [8, 4, 1] / half-res probe grid matches 2560x1440, 1280x720, and 640x360.
        let qhd = D3D12_RESOURCE_DESC { Width: 2560, Height: 1440, ..typeless };
        let half = D3D12_RESOURCE_DESC { Width: 640, Height: 360, ..typeless };
        assert_eq!(score_dispatch_target(&qhd, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [320, 360, 1]), Some(0));
        assert!(score_dispatch_target(&half, DXGI_FORMAT_R16G16B16A16_FLOAT.0, [320, 360, 1]).is_some());

        // CBV V# in a descriptor table decodes back to its buffer VA and byte size.
        let cbv = crate::gpu::desc::cbv_words(0x7_1234_5600, 512);
        assert_eq!(decode_cbv_vsharp(cbv), Some((0x7_1234_5600, 512)));
        let img = crate::gpu::desc::xcl_image_for_view(42, 0, crate::gpu::desc::ImageType::D2, 64, 64, 1, 1);
        assert_eq!(decode_cbv_vsharp([img[0], img[1], img[2], img[3]]), None);

        // Two distinct CPU-only heap slots (`global_index == 0`) at different addresses must not collide
        // before being copied to the shader-visible heap.
        let cpu_slot_a = 0x7F00_0000_0020usize;
        let cpu_slot_b = 0x7F00_0010_0020usize;
        let sv_slot_dst = 0x7F00_0020_0040usize;
        let res_a = 0x7F00_AAAAusize;
        let res_b = 0x7F00_BBBBusize;
        note_descriptor(cpu_slot_a, 0, res_a, DXGI_FORMAT_R16G16B16A16_FLOAT.0, DXGI_FORMAT_R16G16B16A16_TYPELESS.0, true);
        note_descriptor(cpu_slot_b, 0, res_b, DXGI_FORMAT_R8G8B8A8_UNORM.0, DXGI_FORMAT_R8G8B8A8_UNORM.0, false);
        copy_descriptor(cpu_slot_a, 0, sv_slot_dst, 1_100_002);
        {
            let g = VIEW_SLOTS.read().unwrap();
            let vs = g.as_ref().unwrap();
            assert_eq!(vs.by_addr.get(&sv_slot_dst).map(|v| v.res), Some(res_a));
            assert_eq!(vs.by_idx.get(&1_100_002).map(|v| v.res), Some(res_a));
        }
        forget_resource(res_a);
        note_descriptor(cpu_slot_b, 0, 0, 0, 0, false);
    }

    #[test]
    fn scan_cbv_indices_resolves_each_64k_mirror_page_independently() {
        const PAGE: usize = crate::hle::kernelx::mirrors::PAGE;
        // Backing page 0 sits right before a readable decoy slice; backing page 1 is separate.
        let mut backing0_and_decoy = vec![0u32; 128];
        backing0_and_decoy[0] = 101_001; // sentinel in page 0 (last 256 B of GPU page 0)
        backing0_and_decoy[64] = 999_999; // decoy immediately after backing 0 (must NEVER be read)
        let mut backing1 = vec![0u32; 64];
        backing1[0] = 202_002; // sentinel in page 1 (first 256 B of GPU page 1)

        let gpu_page0 = 0x5000_0000usize;
        let gpu_page1 = gpu_page0 + PAGE;
        let cbv_start = gpu_page0 + (PAGE - 256); // 0xff00: spans 256 B on page 0 + 256 B on page 1
        let p0_base = backing0_and_decoy.as_ptr() as usize;
        let p1_base = backing1.as_ptr() as usize;

        let mut seen = Vec::new();
        unsafe {
            scan_cbv_indices_with(
                cbv_start,
                512,
                |va| {
                    if va >= gpu_page0 && va < gpu_page1 {
                        p0_base + (va - cbv_start)
                    } else if va >= gpu_page1 && va < gpu_page1 + 256 {
                        p1_base + (va - gpu_page1)
                    } else {
                        0
                    }
                },
                &mut |v| seen.push(v),
            );
        }
        assert!(seen.contains(&101_001), "first 64 KiB page sentinel must be read");
        assert!(seen.contains(&202_002), "second 64 KiB page sentinel must be resolved independently");
        assert!(!seen.contains(&999_999), "readable decoy past backing 0 must never be read across 64 KiB boundary");
        assert_eq!(resolve_cbv_va(0), 0);
        assert_eq!(resolve_cbv_va(p1_base), p1_base);
    }

    #[test]
    fn ambient_sky_reduce_runs_cleanly_on_gpu() {
        unsafe {
            if std::env::var_os("XCL_D3D12_DEBUG").is_some() {
                let mut dbg: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut dbg).is_ok() {
                    if let Some(d) = dbg {
                        d.EnableDebugLayer();
                    }
                }
            }
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                return;
            }
            let dev = dev.unwrap();
            let dev_raw = dev.as_raw();
            real::adopt(dev_raw, real::Kind::Device);
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).unwrap();
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            let list_raw = list.as_raw();
            real::adopt(list_raw, real::Kind::CommandList);
            list_x::on_create(list_raw, alloc.as_raw() as usize, 0, D3D12_COMMAND_LIST_TYPE_DIRECT);

            // Create a 128x128 RGBA16F render target (sky dome) in RENDER_TARGET state.
            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 128,
                Height: 128,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut rt: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rt_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, None, &mut rt).unwrap();
            let rt = rt.unwrap();

            let ad = build_device(&dev).expect("build_device must succeed");
            let sky_raw = ad.sky_copy.as_raw() as usize;
            let rt0 = rt.as_raw() as usize;
            list_x::transition(list_raw, rt0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list_x::transition(list_raw, sky_raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            list_x::pc_copy_resource(list_raw, sky_raw, rt0);
            list_x::transition(list_raw, sky_raw, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            list_x::transition(list_raw, rt0, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);

            let (cpu0, gpu0) = ad.alloc_group();
            let inc = ad.inc as u64;
            type CreateSrvFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *const D3D12_SHADER_RESOURCE_VIEW_DESC, usize);
            type CreateUavFn = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void, *const D3D12_UNORDERED_ACCESS_VIEW_DESC, usize);
            let create_srv: CreateSrvFn = std::mem::transmute(real::orig_fn(dev_raw, 18));
            let create_uav: CreateUavFn = std::mem::transmute(real::orig_fn(dev_raw, 19));
            let sky_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            };
            let level_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV { FirstElement: 0, NumElements: 1, StructureByteStride: 16, ..Default::default() },
                },
            };
            create_srv(dev_raw, ad.sky_copy.as_raw(), &sky_srv, cpu0 as usize);
            create_uav(dev_raw, ad.sky_level.as_raw(), std::ptr::null_mut(), &level_uav, (cpu0 + inc) as usize);
            create_uav(dev_raw, ad.sky_level.as_raw(), std::ptr::null_mut(), &level_uav, (cpu0 + 2 * inc) as usize);

            let heap_raw = ad.heap.as_raw() as usize;
            let set_heaps: unsafe extern "system" fn(*mut c_void, u32, *const usize) = std::mem::transmute(real::orig_fn(list_raw, 28));
            let set_cs_rs: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list_raw, 29));
            let set_pso: unsafe extern "system" fn(*mut c_void, *mut c_void) = std::mem::transmute(real::orig_fn(list_raw, 25));
            let set_cs_table: unsafe extern "system" fn(*mut c_void, u32, u64) = std::mem::transmute(real::orig_fn(list_raw, 31));
            let set_cs_consts: unsafe extern "system" fn(*mut c_void, u32, u32, *const u32, u32) = std::mem::transmute(real::orig_fn(list_raw, 35));
            let dispatch: unsafe extern "system" fn(*mut c_void, u32, u32, u32) = std::mem::transmute(real::orig_fn(list_raw, 14));
            let consts = [0.22f32.to_bits(), 0.65f32.to_bits(), 0.02f32.to_bits(), 1.0f32.to_bits(), 0, 0, 0, 0];
            set_heaps(list_raw, 1, &heap_raw);
            set_cs_rs(list_raw, ad.rs.as_raw());
            set_pso(list_raw, ad.reduce_pso.as_raw());
            set_cs_consts(list_raw, 0, 8, consts.as_ptr(), 0);
            set_cs_table(list_raw, 1, gpu0);
            set_cs_table(list_raw, 2, gpu0 + inc);
            dispatch(list_raw, 1, 1, 1);
            uav_barrier(list_raw, ad.sky_level.as_raw() as usize);
            list.Close().unwrap();

            let cl: ID3D12CommandList = list.cast().unwrap();
            q.ExecuteCommandLists(&[Some(cl)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap();
            q.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, ev).unwrap();
            windows::Win32::System::Threading::WaitForSingleObject(ev, 5000);
            let _ = windows::Win32::Foundation::CloseHandle(ev);
            assert_eq!(fence.GetCompletedValue(), 1, "ambient sky_reduce GPU dispatch must complete cleanly (device removed: {:?})", dev.GetDeviceRemovedReason());
        }
    }
}
