//! `XCL_DUMP_TARGETS=<work>[,<frames>[,<targets>]]`: what a busy frame draws into. In the first
//! `frames` frames (default 1) with at least `work` draws and dispatches, every render target the
//! frame drew into (and the depth buffer bound most) is logged with how its draws went – ran, drew
//! nothing (a stand-in while the pipeline is built), went to the PC untranslated, skipped – and
//! the `targets` drawn into most (default eight, 64 at most) are written next to the presented
//! frame (`target<k>_<n>.bmp`) with how many of their texels are not zero and how many differ from
//! the first one. A black picture over a scene that was drawn tells a lost composite from a scene
//! that never was; the passes after the scene's (a few draws each) are among the last ranks.
//! `XCL_DUMP_AFTER_S=<s>`: no frame is dumped until `s` seconds after the first present.
//!
//! A draw counts for the frame whose PresentX follows its recording (bundles, and lists executed
//! twice or never, are not exact); an ExecuteIndirect counts once, whatever its count.
//! Multisampled and non-2D targets are logged as not dumped; an array dumps its subresource 0.
//! The depth buffer is not copied: XCL does not know the state the title left it in, and a copy
//! from DEPTH_WRITE (possibly compressed) is invalid – the present list would not close.
//! Targets are kept by address only (not AddRef'd): `present::view_resource` hands one out only
//! while a view still names it, and one released meanwhile is logged as released.
//!
//! Each target written also lists the pipelines that drew into it in that frame (the 12 with
//! most draws): their shader set (`real::shaders_key`), each stage's shader (`real::shader_key`,
//! and the dump hash the `[SHADER]` lines and the debug switches take), render target 0's blend
//! state and the formats as the title made them, and what the first draw with it read through
//! its descriptor tables that is not a target written. The console VS and PS of each are kept
//! when the pipeline is made (512 MiB of them at most) and translated again on a thread of their
//! own after the dump: `target<k>_<stage>_<key>.hlsl` and `.rdna2.txt` (the disassembly)
//! go next to the pictures. Pipelines are known only once made through
//! CreateGraphicsPipelineState (or a stream of console VS / PS) with the switch on.
//! The switch also follows 3D textures and reads the small ones back at a dump (`lut3d`).
//! The dump also lists every compute pipeline the frame dispatched, in order of first use: its
//! CS, its direct and indirect dispatches, the groups of the first direct one and of the largest,
//! and the first indirect one's arguments when they were in unified memory when it was recorded.
//! The first `COMPUTE_FILES_MOST` get `compute<k>_cs_<key>.hlsl` and `.rdna2.txt` the same way
//! (made through CreateComputePipelineState with the switch on; the HLSL even when the pipeline
//! runs direct DXIL, whose file names the path the running DXIL took).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// targets dumped of one frame when the switch names no number, and the most it may name
const DUMPED: usize = 8;
const DUMPED_MOST: usize = 64;

/// How a draw went: ran on its translated pipeline, drew nothing (a stand-in for a pipeline still
/// being built), went to the PC untranslated, or was skipped.
pub const RAN: usize = 0;
pub const STANDIN: usize = 1;
pub const PC: usize = 2;
pub const SKIPPED: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Config {
    /// draws and dispatches a frame needs
    work: u64,
    /// frames to dump
    frames: u64,
    /// no dump before this long after the first present
    after: Duration,
    /// targets written of a frame
    targets: usize,
}

fn config() -> Option<Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    *C.get_or_init(|| {
        let v = std::env::var("XCL_DUMP_TARGETS").ok()?;
        let Some((work, frames, targets)) = parse(&v) else {
            crate::xlog_warn!("D3D12.X: XCL_DUMP_TARGETS={v:?} is not <draws and dispatches>[,<frames>[,<targets>]]: no targets dumped");
            return None;
        };
        let after = match std::env::var("XCL_DUMP_AFTER_S") {
            Ok(s) => parse_after(&s).unwrap_or_else(|| {
                crate::xlog_warn!("D3D12.X: XCL_DUMP_AFTER_S={s:?} is not a number of seconds: dumps start at once");
                Duration::ZERO
            }),
            Err(_) => Duration::ZERO,
        };
        Some(Config { work, frames, after, targets })
    })
}

/// (work, frames, targets written of a frame: 1 ..= `DUMPED_MOST`)
fn parse(v: &str) -> Option<(u64, u64, usize)> {
    let mut it = v.split(',').map(|s| s.trim().parse::<u64>());
    let work = it.next()?.ok()?;
    let frames = match it.next() {
        Some(n) => n.ok()?,
        None => 1,
    };
    let targets = match it.next() {
        Some(n) => n.ok()? as usize,
        None => DUMPED,
    };
    (work > 0 && frames > 0 && targets > 0 && it.next().is_none()).then_some((work, frames, targets.min(DUMPED_MOST)))
}

/// `XCL_DUMP_AFTER_S`: seconds, fractions too (None: not a number of seconds)
fn parse_after(v: &str) -> Option<Duration> {
    v.trim().parse::<f64>().ok().and_then(|s| Duration::try_from_secs_f64(s).ok())
}

pub fn enabled() -> bool {
    config().is_some()
}

/// What a frame drew: its draws and dispatches, and the draws into each target by outcome.
#[derive(Default, Debug, PartialEq)]
pub struct Frame {
    work: u64,
    targets: HashMap<usize, [u64; 4]>,
    depth: HashMap<usize, [u64; 4]>,
    /// draws by (target, pipeline)
    by: HashMap<(usize, usize), u64>,
    /// what the first draw with each pipeline read (`bind::table_resources`)
    inputs: HashMap<usize, Vec<Input>>,
    /// the compute pipelines dispatched, in order of first use, and where each is in it
    computes: Vec<Compute>,
    compute_at: HashMap<usize, usize>,
}

/// A dispatch: a direct one's groups, or an indirect one's arguments as read when it was
/// recorded (None: not in memory the CPU reads).
pub enum Dispatched {
    Direct([u32; 3]),
    Indirect(Option<String>),
}

/// A compute pipeline's dispatches in a frame.
#[derive(Clone, Debug, Default, PartialEq)]
struct Compute {
    pso: usize,
    direct: u64,
    indirect: u64,
    /// groups of the first direct dispatch, and of the one with most groups
    first: Option<[u32; 3]>,
    largest: Option<[u32; 3]>,
    /// the first indirect one's arguments (`Dispatched::Indirect`)
    first_indirect: Option<Option<String>>,
}

/// A resource a draw read through its descriptor tables, described when it drew.
#[derive(Clone, Debug, PartialEq)]
struct Input {
    res: usize,
    /// D3D12_RESOURCE_DIMENSION; width, height, depth or array size, mips, format
    dimension: i32,
    width: u64,
    height: u32,
    depth: u16,
    mips: u16,
    format: i32,
}

impl Frame {
    /// Draws with pipeline `pso` into `targets`: true for the frame's first draw with it (its
    /// inputs are noted once).
    fn drew_with(&mut self, targets: &[usize], pso: usize) -> bool {
        for &t in targets {
            *self.by.entry((t, pso)).or_default() += 1;
        }
        let first = !self.inputs.contains_key(&pso);
        self.inputs.entry(pso).or_default();
        first
    }

    /// A dispatch with compute pipeline `pso`.
    fn dispatched(&mut self, pso: usize, d: Dispatched) {
        let k = *self.compute_at.entry(pso).or_insert_with(|| {
            self.computes.push(Compute { pso, ..Default::default() });
            self.computes.len() - 1
        });
        let c = &mut self.computes[k];
        match d {
            Dispatched::Direct(g) => {
                let groups = |g: [u32; 3]| g.iter().map(|&n| n as u64).product::<u64>();
                c.direct += 1;
                c.first.get_or_insert(g);
                if c.largest.map_or(true, |l| groups(g) > groups(l)) {
                    c.largest = Some(g);
                }
            }
            Dispatched::Indirect(a) => {
                c.indirect += 1;
                c.first_indirect.get_or_insert(a);
            }
        }
    }

    /// The pipelines that drew into `target`, most draws first (ties: by address), the first
    /// `most` of them, and how many more there were.
    fn pipelines_of(&self, target: usize, most: usize) -> (Vec<(usize, u64)>, usize) {
        let mut v: Vec<(usize, u64)> = self.by.iter().filter(|((t, _), _)| *t == target).map(|(&(_, p), &n)| (p, n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let more = v.len().saturating_sub(most);
        v.truncate(most);
        (v, more)
    }

    fn draw(&mut self, targets: &[usize], depth: Option<usize>, outcome: usize) {
        self.work += 1;
        for &t in targets {
            self.targets.entry(t).or_default()[outcome] += 1;
        }
        if let Some(d) = depth {
            self.depth.entry(d).or_default()[outcome] += 1;
        }
    }

    /// targets by draws into them, most first (ties: by address, for a stable order)
    fn ranked(&self) -> Vec<(usize, [u64; 4])> {
        let mut v: Vec<(usize, [u64; 4])> = self.targets.iter().map(|(&t, &n)| (t, n)).collect();
        v.sort_by(|a, b| b.1.iter().sum::<u64>().cmp(&a.1.iter().sum()).then(a.0.cmp(&b.0)));
        v
    }

    /// the depth buffer drawn with most
    fn depth(&self) -> Option<(usize, [u64; 4])> {
        self.depth.iter().map(|(&d, &n)| (d, n)).max_by(|a, b| a.1.iter().sum::<u64>().cmp(&b.1.iter().sum()).then(b.0.cmp(&a.0)))
    }
}

/// the frame being recorded
static FRAME: Mutex<Option<Frame>> = Mutex::new(None);
/// frames dumped so far (their copies submitted)
static DONE: AtomicU64 = AtomicU64::new(0);
/// the frame to dump at the next present that gets as far as copying
static PENDING: Mutex<Option<Frame>> = Mutex::new(None);

/// A draw on `list` with `outcome`: it counts for each render target and the depth buffer bound.
pub unsafe fn note_draw(list: *mut c_void, outcome: usize) {
    if !enabled() {
        return;
    }
    let mut targets: Vec<usize> = super::list_x::bound_render_targets(list).into_iter().filter_map(super::present::rtv_resource).collect();
    targets.sort_unstable();
    targets.dedup();
    for &t in &targets {
        super::lut3d::wrote(t, false, || format!("drawn into as a render target (pipeline {:#x})", super::bind::bound_pso(list)));
    }
    let depth = super::list_x::bound_dsv(list).and_then(super::present::rtv_resource);
    let pso = super::bind::bound_pso(list);
    let first = {
        let mut g = FRAME.lock().unwrap();
        let f = g.get_or_insert_with(Frame::default);
        f.draw(&targets, depth, outcome);
        f.drew_with(&targets, pso)
    };
    if first {
        // (described now: the title draws with them, so they are alive)
        let inputs: Vec<Input> = super::bind::table_resources(list)
            .into_iter()
            .filter_map(|r| {
                let d = ID3D12Resource::from_raw_borrowed(&(r as *mut c_void))?.GetDesc();
                Some(Input { res: r, dimension: d.Dimension.0, width: d.Width, height: d.Height, depth: d.DepthOrArraySize, mips: d.MipLevels, format: d.Format.0 })
            })
            .collect();
        FRAME.lock().unwrap().get_or_insert_with(Frame::default).inputs.insert(pso, inputs);
    }
}

/// A dispatch: work of the frame.
pub fn note_dispatch() {
    if enabled() {
        FRAME.lock().unwrap().get_or_insert_with(Frame::default).work += 1;
    }
}

/// A dispatch with compute pipeline `pso` (the object the title bound): listed in the dump.
pub fn note_compute(pso: usize, d: Dispatched) {
    if enabled() {
        FRAME.lock().unwrap().get_or_insert_with(Frame::default).dispatched(pso, d);
    }
}

/// An ExecuteIndirect of up to `max` dispatches with `pso`: its first command at `commands`
/// (address, bytes there) and its count at `count` when they are in unified memory, read now
/// (what the title wrote there by the time it recorded); `plain`: a signature of the dispatch
/// argument alone; `counted`: it has a count buffer.
pub unsafe fn note_indirect_compute(pso: usize, commands: Option<(usize, usize)>, count: Option<usize>, counted: bool, max: u32, plain: bool) {
    if !enabled() {
        return;
    }
    let count = match count.filter(|&p| crate::gpu::shader::readable(p, 4)) {
        Some(p) => format!(", count {}", (p as *const u32).read_unaligned()),
        None if counted => ", count on the GPU".into(),
        None => String::new(),
    };
    let words = |p: usize, n: usize| (0..n).map(|k| ((p + 4 * k) as *const u32).read_unaligned()).collect::<Vec<u32>>();
    let args = match commands {
        Some((p, n)) if plain && n >= 12 && crate::gpu::shader::readable(p, 12) => {
            let w = words(p, 3);
            Some(format!("arguments {}x{}x{} (max {max}{count})", w[0], w[1], w[2]))
        }
        Some((p, n)) if !plain && n >= 4 && crate::gpu::shader::readable(p, n.min(32)) => {
            Some(format!("a command with root arguments, its first dwords {:x?} (max {max}{count})", words(p, n.min(32) / 4)))
        }
        _ => None,
    };
    note_compute(pso, Dispatched::Indirect(args));
}

/// Whether a frame of `work` is dumped, `done` frames dumped before it, `since` the first present.
fn due(c: &Config, work: u64, done: u64, since: Duration) -> bool {
    work >= c.work && done < c.frames && since >= c.after
}

/// The title presents: a frame that did enough work is dumped at the next present that copies
/// (one still waiting for that is kept, and this one dropped).
pub fn frame_end() {
    let Some(c) = config() else { return };
    static T0: OnceLock<Instant> = OnceLock::new();
    let t0 = *T0.get_or_init(Instant::now);
    let f = FRAME.lock().unwrap().take().unwrap_or_default();
    if !due(&c, f.work, DONE.load(Ordering::Relaxed), t0.elapsed()) {
        return;
    }
    PENDING.lock().unwrap().get_or_insert(f);
}

/// The frame to dump at this present, if any (`submitted` once its copies are on the GPU).
pub fn take_pending() -> Option<Frame> {
    if !enabled() {
        return None;
    }
    PENDING.lock().unwrap().take()
}

/// The copies of a frame taken by `take_pending` were submitted: it counts as dumped.
pub fn submitted() {
    DONE.fetch_add(1, Ordering::Relaxed);
}

/// Logs what frame `n` drew into; the targets to copy for its dump: (tag, resource, desc).
pub unsafe fn to_dump(n: u64, f: &Frame) -> Vec<(String, ID3D12Resource, D3D12_RESOURCE_DESC)> {
    let ranked = f.ranked();
    let dumped = config().map_or(DUMPED, |c| c.targets);
    crate::xlog_info!("D3D12.X: target dump of frame {n}: {} draws and dispatches, {} render targets drawn into", f.work, ranked.len());
    let mut out = Vec::new();
    let (mut sizes, mut left, mut written) = (Vec::new(), Vec::new(), Vec::new());
    let depth = f.depth().map(|(r, c)| (usize::MAX, r, c));
    for (k, r, c) in ranked.into_iter().enumerate().map(|(k, (r, c))| (k, r, c)).chain(depth) {
        let res = super::present::view_resource(r);
        let d = res.as_ref().map(|x| x.GetDesc());
        let what = d.map_or("released".into(), |d| format!("{}x{} format {}", d.Width, d.Height, d.Format.0));
        let copyable = d.is_some_and(|d| d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.SampleDesc.Count <= 1);
        // past the readback cap: this target and every later one are left out
        let capped = k < dumped && copyable && {
            sizes.push(d.map_or(0, |d| readback_bytes(&d)));
            dumped_within(&sizes, DUMP_BYTES_MOST) < sizes.len()
        };
        let tag = match (k, copyable) {
            (usize::MAX, _) => " (not dumped: its state is not known)".into(),
            (k, true) if k < dumped && capped => format!(" (not dumped: target{k} is past the readback cap)"),
            (k, true) if k < dumped => format!(" → target{k}"),
            (_, false) if d.is_some() => " (not dumped)".into(),
            _ => String::new(),
        };
        crate::xlog_info!(
            "D3D12.X: target dump:   {} {r:#x} {what}: {} draws (ran {}, stand-in {}, PC {}, skipped {}){tag}",
            if k == usize::MAX { "depth buffer" } else { "render target" },
            c.iter().sum::<u64>(),
            c[RAN],
            c[STANDIN],
            c[PC],
            c[SKIPPED]
        );
        if let (true, Some(res), Some(d)) = (k < dumped && copyable, res, d) {
            if capped {
                left.push(format!("target{k}"));
            } else {
                out.push((format!("target{k}"), res, d));
                written.push((k, r));
            }
        }
    }
    if !left.is_empty() {
        crate::xlog_warn!("D3D12.X: target dump of frame {n}: {} of {} targets left out, past {} MiB of readback: {}", left.len(), out.len() + left.len(), DUMP_BYTES_MOST >> 20, left.join(" "));
    }
    let mut files = log_pipelines(n, f, &written);
    files.extend(log_computes(n, f));
    if !files.is_empty() {
        if let Some(dir) = crate::state::boot().map(|b| b.logs_dir()) {
            let _ = std::thread::Builder::new().name("xcl-dump-shaders".into()).spawn(move || {
                write_shaders(&dir, files, HLSL_BYTES_MOST);
                crate::xlog_info!("D3D12.X: target dump: the shaders of the pipelines listed are written to {}", dir.display());
            });
        }
    }
    out
}

/// compute pipelines of a frame whose shaders a dump writes at most
const COMPUTE_FILES_MOST: usize = 400;

/// The files of a frame's `k`th compute pipeline's shader: `compute<k>_cs_<key>` (.hlsl, .rdna2.txt).
fn compute_stem(k: usize, key: u64) -> String {
    format!("compute{k}_cs_{key:016x}")
}

/// `x`x`y`x`z`, or "none"
fn groups_text(g: Option<[u32; 3]>) -> String {
    g.map_or("none".into(), |g| format!("{}x{}x{}", g[0], g[1], g[2]))
}

/// Logs the compute pipelines frame `n` dispatched, in order of first use; the shader files of
/// the first `COMPUTE_FILES_MOST` of them.
fn log_computes(n: u64, f: &Frame) -> Vec<ShaderFile> {
    let mut files = Vec::new();
    if f.computes.is_empty() {
        return files;
    }
    crate::xlog_info!("D3D12.X: target dump of frame {n}: {} compute pipelines dispatched, in order of first use", f.computes.len());
    let pipelines = PIPELINES.lock().unwrap().clone().unwrap_or_default();
    for (k, c) in f.computes.iter().enumerate() {
        let translated = super::real::is_translated_compute(super::pso_async::peek(c.pso));
        let cs = pipelines.get(&c.pso).and_then(|p| p.stages.iter().find(|s| s.0 == "cs").map(|s| (s.1, s.2, s.3.clone(), p.split)));
        let shader = cs.as_ref().map_or("cs not known (made before the switch)".into(), |(key, h, ..)| format!("cs {key:016x} (dump hash {h:016x})"));
        let indirect = match &c.first_indirect {
            None => String::new(),
            Some(Some(a)) => format!("; the first indirect one: {a}"),
            Some(None) => "; the first indirect one: arguments on the GPU".into(),
        };
        crate::xlog_info!(
            "D3D12.X: target dump: compute #{k} pipeline {:#x}: {shader}, {} dispatches ({} direct, {} indirect), groups of the first direct one {}, largest {}{indirect}; translated {translated}",
            c.pso,
            c.direct + c.indirect,
            c.direct,
            c.indirect,
            groups_text(c.first),
            groups_text(c.largest)
        );
        if let (true, Some((key, _, Some(b), split))) = (k < COMPUTE_FILES_MOST, cs) {
            files.push((compute_stem(k, key), "cs", b, None, split));
        }
    }
    if f.computes.len() > COMPUTE_FILES_MOST {
        crate::xlog_info!("D3D12.X: target dump of frame {n}: the shaders of compute #{COMPUTE_FILES_MOST} and later are not written ({COMPUTE_FILES_MOST} pipelines at most)");
    }
    files
}

/// pipelines a written target lists
const PIPELINES_MOST: usize = 48;
/// console shader bytes kept for the dumps at most
const BLOB_BYTES_MOST: usize = 512 << 20;

/// A graphics pipeline as the title made it.
struct Pipeline {
    /// `real::shaders_key` of its shaders
    set: u64,
    /// (stage, `real::shader_key`, dump hash, its bytes: None past `BLOB_BYTES_MOST`)
    stages: Vec<(&'static str, u64, u64, Option<Arc<[u8]>>)>,
    blend: D3D12_RENDER_TARGET_BLEND_DESC,
    formats: Vec<i32>,
    /// VS / PS translated for waves of 32 (`real::graphics_wave32`)
    split: bool,
}

/// pipelines by the address the title has (kept by address: a later one there takes its place)
static PIPELINES: Mutex<Option<HashMap<usize, Arc<Pipeline>>>> = Mutex::new(None);
/// console shaders by `real::shader_key`, and their bytes in all
static BLOBS: Mutex<Option<(HashMap<u64, Arc<[u8]>>, usize)>> = Mutex::new(None);

/// Pipeline `pso` was made of `stages` (VS, PS, DS, HS, GS as present, in that order) with render
/// target 0's `blend` and the render target `formats`.
pub fn note_pipeline(pso: usize, stages: &[(&'static str, &[u8])], blend: &D3D12_RENDER_TARGET_BLEND_DESC, formats: &[DXGI_FORMAT], split: bool) {
    if !enabled() || pso == 0 {
        return;
    }
    let set = super::real::shaders_key(&stages.iter().map(|(_, b)| b.to_vec()).collect::<Vec<_>>());
    let stages = stages
        .iter()
        .map(|&(s, b)| {
            let (key, hash, blob) = keep_shader(b);
            (s, key, hash, blob)
        })
        .collect();
    let p = Pipeline { set, stages, blend: *blend, formats: formats.iter().map(|f| f.0).collect(), split };
    PIPELINES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, Arc::new(p));
}

/// Shader `b` kept for a dump (`XCL_DUMP_TARGETS`, `XCL_CAPTURE_FRAME`; 512 MiB of them at most):
/// its `real::shader_key`, its dump hash, and its bytes (None past the cap).
pub(super) fn keep_shader(b: &[u8]) -> (u64, u64, Option<Arc<[u8]>>) {
    let key = super::real::shader_key(b);
    let mut g = BLOBS.lock().unwrap();
    let (kept, bytes) = g.get_or_insert_with(Default::default);
    let blob = kept.get(&key).cloned().or_else(|| {
        if *bytes + b.len() > BLOB_BYTES_MOST {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::xlog_warn!("D3D12.X: the dump switches keep {} MiB of shaders: later ones get no HLSL in a dump", BLOB_BYTES_MOST >> 20));
            return None;
        }
        *bytes += b.len();
        let a: Arc<[u8]> = b.into();
        kept.insert(key, a.clone());
        Some(a)
    });
    (key, crate::gpu::shader::fnv64(b), blob)
}

/// A D3D12_BLEND by name.
fn blend_name(b: i32) -> &'static str {
    const NAMES: [&str; 22] = [
        "?", "ZERO", "ONE", "SRC_COLOR", "INV_SRC_COLOR", "SRC_ALPHA", "INV_SRC_ALPHA", "DEST_ALPHA", "INV_DEST_ALPHA", "DEST_COLOR", "INV_DEST_COLOR", "SRC_ALPHA_SAT", "?", "?",
        "BLEND_FACTOR", "INV_BLEND_FACTOR", "SRC1_COLOR", "INV_SRC1_COLOR", "SRC1_ALPHA", "INV_SRC1_ALPHA", "ALPHA_FACTOR", "INV_ALPHA_FACTOR",
    ];
    NAMES.get(b as usize).copied().unwrap_or("?")
}

/// Render target 0's blend state as a line: what a pass that comes out inverted is checked by.
pub(super) fn blend_text(b: &D3D12_RENDER_TARGET_BLEND_DESC) -> String {
    const OPS: [&str; 6] = ["?", "ADD", "SUBTRACT", "REV_SUBTRACT", "MIN", "MAX"];
    const LOGIC: [&str; 16] = [
        "CLEAR", "SET", "COPY", "COPY_INVERTED", "NOOP", "INVERT", "AND", "NAND", "OR", "NOR", "XOR", "EQUIV", "AND_REVERSE", "AND_INVERTED", "OR_REVERSE", "OR_INVERTED",
    ];
    let op = |o: D3D12_BLEND_OP| OPS.get(o.0 as usize).copied().unwrap_or("?");
    let mut s = if b.BlendEnable.as_bool() {
        format!(
            "blend on: colour {} {} {}, alpha {} {} {}",
            blend_name(b.SrcBlend.0),
            op(b.BlendOp),
            blend_name(b.DestBlend.0),
            blend_name(b.SrcBlendAlpha.0),
            op(b.BlendOpAlpha),
            blend_name(b.DestBlendAlpha.0)
        )
    } else {
        "blend off".to_string()
    };
    if b.LogicOpEnable.as_bool() {
        s += &format!(", logic op {}", LOGIC.get(b.LogicOp.0 as usize).copied().unwrap_or("?"));
    }
    s + &format!(", write mask {:#x}", b.RenderTargetWriteMask)
}

/// The files of a written target's pipeline's shader: `target<k>_<stage>_<key>` (.hlsl, .rdna2.txt).
fn shader_stem(k: usize, stage: &str, key: u64) -> String {
    format!("target{k}_{stage}_{key:016x}")
}

/// Logs the pipelines that drew into each written target `(k, resource)` of frame `n`; the files
/// of their shaders, written on a thread of their own (a later frame's dump writes over them).
fn log_pipelines(n: u64, f: &Frame, written: &[(usize, usize)]) -> Vec<ShaderFile> {
    crate::xlog_info!("D3D12.X: target dump of frame {n}: the pipelines of the targets written");
    let pipelines = PIPELINES.lock().unwrap().clone().unwrap_or_default();
    let tag_of = |r: usize| written.iter().find(|w| w.1 == r).map(|w| format!("target{}", w.0));
    let mut files: Vec<ShaderFile> = Vec::new();
    for &(k, t) in written {
        let (list, more) = f.pipelines_of(t, PIPELINES_MOST);
        for (pso, draws) in list {
            let translated = super::real::is_translated_graphics(super::pso_async::peek(pso));
            let Some(p) = pipelines.get(&pso) else {
                crate::xlog_info!("D3D12.X: target dump: target{k}: {draws} draws with pipeline {pso:#x} (translated {translated}): not known (made before the switch, or a mesh pipeline)");
                continue;
            };
            let stages: Vec<String> = p.stages.iter().map(|(s, key, h, _)| format!("{s} {key:016x} (dump hash {h:016x})")).collect();
            let inputs = f.inputs.get(&pso).map_or(&[][..], |v| &v[..]);
            let read: Vec<String> = inputs.iter().filter_map(|i| tag_of(i.res)).collect();
            crate::xlog_info!(
                "D3D12.X: target dump: target{k}: {draws} draws with pipeline {pso:#x} (translated {translated}): shaders {:016x} – {}; RT formats {:?}, RT0 {}{}",
                p.set,
                stages.join(", "),
                p.formats,
                blend_text(&p.blend),
                if read.is_empty() { String::new() } else { format!("; reads {}", read.join(" ")) }
            );
            for i in inputs.iter().filter(|i| tag_of(i.res).is_none()) {
                crate::xlog_info!(
                    "D3D12.X: target dump: target{k}:   pipeline {pso:#x} reads {:#x}: dimension {}, {}x{}x{}, {} mips, format {}",
                    i.res,
                    i.dimension,
                    i.width,
                    i.height,
                    i.depth,
                    i.mips,
                    i.format
                );
            }
            let vs = p.stages.iter().find(|s| s.0 == "vs").and_then(|s| s.3.clone());
            for (s, key, _, blob) in &p.stages {
                if let Some(b) = blob {
                    files.push((shader_stem(k, s, *key), s, b.clone(), vs.clone(), p.split));
                }
            }
        }
        if more > 0 {
            crate::xlog_info!("D3D12.X: target dump: target{k}: and {more} more pipelines");
        }
    }
    files
}

/// A shader to write (`write_shaders`): its file stem, stage (`vs`, `ps`, `cs`, …), bytes, the
/// pipeline's VS (a PS's static samplers come after the VS's) and whether it is translated for
/// waves of 32.
pub(super) type ShaderFile = (String, &'static str, Arc<[u8]>, Option<Arc<[u8]>>, bool);

/// HLSL bytes of one shader a dump writes at most (`XCL_DUMP_TARGETS`, `XCL_CAPTURE_FRAME`; a
/// retail title's biggest programs are tens of MB of it)
pub(super) const HLSL_BYTES_MOST: usize = 8 << 20;

/// Writes each console shader's disassembly and its VS / PS / CS translated again, as the pipeline
/// translates it through HLSL (`pso::console_translation`; the PS's static samplers after the
/// VS's). HLSL past `hlsl_most` bytes is not written: a note of its size takes its place. A CS's
/// file says which path made the DXIL the pipeline runs (`pso::compute_path`).
pub(super) fn write_shaders(dir: &std::path::Path, files: Vec<ShaderFile>, hlsl_most: usize) {
    use crate::gpu::recomp::pso::{console_translation, is_console, mem_windows, Stage};
    let translate = |b: &[u8], stage: Stage, base: u32, split: bool| console_translation(b, stage, base, mem_windows(), split).map(|(t, _)| (t.hlsl, t.static_samplers.len() as u32));
    for (stem, stage, blob, vs, split) in files {
        let base = dir.join(&stem);
        if !is_console(&blob) {
            let _ = std::fs::write(base.with_extension("txt"), format!("not a console shader ({} bytes): the PC compiles it as it is\n", blob.len()));
            continue;
        }
        if let Some(sh) = crate::gpu::xbox::parse(&blob) {
            let _ = std::fs::write(base.with_extension("rdna2.txt"), crate::gpu::xbox::listing(&sh));
        }
        let hlsl = match stage {
            "vs" => translate(&blob, Stage::Vertex, 0, split),
            "ps" => {
                let samplers = vs.as_deref().filter(|v| is_console(v)).and_then(|v| translate(v, Stage::Vertex, 0, split).ok()).map_or(0, |t| t.1);
                translate(&blob, Stage::Pixel, samplers, split)
            }
            // (for the wave size the pipeline was noted with)
            "cs" => translate(&blob, Stage::Compute, 0, split),
            _ => Err(format!("a {stage} is not translated here")),
        };
        let text = match hlsl {
            Ok((h, _)) if h.len() > hlsl_most => format!("// {stem}: translated, {} bytes of HLSL: past {} MiB, not written\n", h.len(), hlsl_most >> 20),
            Ok((h, _)) if stage == "cs" => format!(
                "// {stem}: translated again from the pipeline's console shader through HLSL, for reading (a switch-form retry or a debug switch build another)\n// the running pipeline's DXIL: {}\n{h}",
                crate::gpu::recomp::pso::compute_path(crate::gpu::shader::fnv64(&blob)).unwrap_or("not known (not translated in this process yet)")
            ),
            Ok((h, _)) => format!("// {stem}: translated again from the pipeline's console shader (the HLSL path; XCL_DIRECT_DXIL, a switch-form retry or a debug switch build another)\n{h}"),
            Err(e) => format!("// {stem}: not translated: {e}\n"),
        };
        let _ = std::fs::write(base.with_extension("hlsl"), text);
    }
}

/// readback bytes of one present's target dump at most: 64 targets of 2560x1440 RGBA32F are
/// about 3.8 GB, at once, on a card that may be full already
const DUMP_BYTES_MOST: u64 = 1 << 30;

/// bytes a copy of `d`'s subresource 0 reads back (row padding aside)
fn readback_bytes(d: &D3D12_RESOURCE_DESC) -> u64 {
    d.Width.saturating_mul(d.Height as u64).saturating_mul(texel_bytes(d.Format) as u64)
}

/// Of targets of `sizes` bytes, in the order they are dumped: how many go before `most` bytes
/// are passed (the rest are left out). The first goes however large it is.
fn dumped_within(sizes: &[u64], most: u64) -> usize {
    let mut total = 0u64;
    for (k, &b) in sizes.iter().enumerate() {
        total = total.saturating_add(b);
        if k > 0 && total > most {
            return k;
        }
    }
    sizes.len()
}

/// Of the texels sampled in `data` (every 4th of every 4th row, as the dumps do): how many are not
/// all zero, how many differ from the first one sampled (a cleared target has none), and how many
/// were sampled; any format, by its texel size; texels past the end of `data` are not sampled.
fn census(data: &[u8], width: usize, height: usize, pitch: usize, format: DXGI_FORMAT) -> (u64, u64, u64) {
    let bytes = texel_bytes(format);
    let (mut lit, mut differ, mut n) = (0, 0, 0);
    let mut first: Option<&[u8]> = None;
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let at = y * pitch + x * bytes;
            let Some(t) = data.get(at..at + bytes) else { continue };
            n += 1;
            if t.iter().any(|&b| b != 0) {
                lit += 1;
            }
            if *first.get_or_insert(t) != t {
                differ += 1;
            }
        }
    }
    (lit, differ, n)
}

/// bytes of a texel of a render-target format (or of the plane of a planar format a footprint
/// names: a copied depth-stencil's depth plane is R32 / R24X8 typeless, 4 bytes)
pub(super) fn texel_bytes(f: DXGI_FORMAT) -> usize {
    match f {
        DXGI_FORMAT_R32G32B32A32_FLOAT | DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT | DXGI_FORMAT_R32G32B32A32_TYPELESS => 16,
        DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_UNORM | DXGI_FORMAT_R16G16B16A16_UINT | DXGI_FORMAT_R16G16B16A16_SNORM
        | DXGI_FORMAT_R16G16B16A16_SINT | DXGI_FORMAT_R16G16B16A16_TYPELESS | DXGI_FORMAT_R32G32_FLOAT | DXGI_FORMAT_R32G32_UINT
        | DXGI_FORMAT_R32G32_SINT | DXGI_FORMAT_R32G32_TYPELESS | DXGI_FORMAT_D32_FLOAT_S8X24_UINT | DXGI_FORMAT_R32G8X24_TYPELESS => 8,
        DXGI_FORMAT_R8G8_UNORM | DXGI_FORMAT_R8G8_UINT | DXGI_FORMAT_R8G8_SNORM | DXGI_FORMAT_R8G8_SINT | DXGI_FORMAT_R8G8_TYPELESS
        | DXGI_FORMAT_R16_FLOAT | DXGI_FORMAT_R16_UNORM | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SNORM | DXGI_FORMAT_R16_SINT
        | DXGI_FORMAT_R16_TYPELESS | DXGI_FORMAT_D16_UNORM | DXGI_FORMAT_B5G6R5_UNORM | DXGI_FORMAT_B5G5R5A1_UNORM | DXGI_FORMAT_B4G4R4A4_UNORM => 2,
        DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_SNORM | DXGI_FORMAT_R8_SINT | DXGI_FORMAT_A8_UNORM | DXGI_FORMAT_R8_TYPELESS => 1,
        // also D32 / R32 typeless, D24S8 / R24G8 typeless, R11G11B10
        _ => 4,
    }
}

/// A dump `tag` read back: how many of its texels are not all zero and differ from the first.
/// The texel size is that of the footprint's format – the copied plane's for a planar format,
/// not the resource's – and the bytes read stop at the buffer's end.
pub unsafe fn log_census(rb: &ID3D12Resource, fp: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, tag: &str) {
    let mut p: *mut c_void = std::ptr::null_mut();
    if rb.Map(0, None, Some(&mut p)).is_err() || p.is_null() {
        return;
    }
    let f = &fp.Footprint;
    let len = (rb.GetDesc().Width as usize).saturating_sub(fp.Offset as usize);
    let data = std::slice::from_raw_parts((p as *const u8).add(fp.Offset as usize), len);
    let (lit, differ, n) = census(data, f.Width as usize, f.Height as usize, f.RowPitch as usize, f.Format);
    rb.Unmap(0, None);
    crate::xlog_info!("D3D12.X: target dump: {tag}: {lit} of {n} texels sampled not zero, {differ} differ from the first texel (format {})", f.Format.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_takes_a_work_threshold_and_a_frame_count() {
        assert_eq!(parse("200"), Some((200, 1, DUMPED)));
        assert_eq!(parse("200,3"), Some((200, 3, DUMPED)));
        assert_eq!(parse(" 50 , 2 "), Some((50, 2, DUMPED)));
        assert_eq!(parse("0"), None);
        assert_eq!(parse("200,0"), None);
        assert_eq!(parse("x"), None);
        assert_eq!(parse("200,x"), None);
        // a third number: the targets written of a frame, 64 at most
        assert_eq!(parse("200,3,24"), Some((200, 3, 24)));
        assert_eq!(parse("200,3,1000"), Some((200, 3, DUMPED_MOST)));
        assert_eq!(parse("200,3,0"), None);
        assert_eq!(parse("200,3,x"), None);
        assert_eq!(parse("200,3,24,1"), None);
    }

    #[test]
    fn dumps_wait_the_seconds_after_the_first_present() {
        assert_eq!(parse_after(" 30 "), Some(Duration::from_secs(30)));
        assert_eq!(parse_after("2.5"), Some(Duration::from_millis(2500)));
        assert_eq!(parse_after("soon"), None);
        assert_eq!(parse_after("-1"), None);
        assert_eq!(parse_after("inf"), None);
        let c = Config { work: 100, frames: 2, after: Duration::from_secs(30), targets: DUMPED };
        assert!(!due(&c, 500, 0, Duration::from_secs(29)), "too early");
        assert!(due(&c, 500, 0, Duration::from_secs(30)));
        assert!(!due(&c, 99, 0, Duration::from_secs(40)), "too little work");
        assert!(due(&c, 100, 1, Duration::from_secs(40)));
        assert!(!due(&c, 100, 2, Duration::from_secs(40)), "frames all dumped");
    }

    #[test]
    fn draws_count_per_target_by_outcome() {
        let mut f = Frame::default();
        f.draw(&[0x10, 0x20], Some(0x90), RAN);
        f.draw(&[0x10], Some(0x90), STANDIN);
        f.draw(&[0x10], None, PC);
        f.draw(&[0x20], Some(0x80), SKIPPED);
        f.draw(&[], None, RAN);
        assert_eq!(f.work, 5);
        assert_eq!(f.targets[&0x10], [1, 1, 1, 0]);
        assert_eq!(f.targets[&0x20], [1, 0, 0, 1]);
        assert_eq!(f.depth(), Some((0x90, [1, 1, 0, 0])));
        assert_eq!(Frame::default().depth(), None);
    }

    #[test]
    fn targets_rank_by_draws_and_texels_count_by_format() {
        let mut f = Frame::default();
        for (t, n) in [(0x30, 5), (0x10, 9), (0x20, 5)] {
            for _ in 0..n {
                f.draw(&[t], None, if n == 9 { SKIPPED } else { RAN });
            }
        }
        let order: Vec<(usize, u64)> = f.ranked().into_iter().map(|(t, c)| (t, c.iter().sum())).collect();
        assert_eq!(order, vec![(0x10, 9), (0x20, 5), (0x30, 5)]);
        // 8 x 8 texels of 8 bytes (R16G16B16A16_FLOAT), every 4th of every 4th row sampled:
        // (0,0) (4,0) (0,4) (4,4); one of them has a non-zero byte in its last half
        let (w, h, bytes) = (8usize, 8usize, 8usize);
        let pitch = w * bytes;
        let mut data = vec![0u8; pitch * h];
        data[4 * pitch + 4 * bytes + 7] = 1;
        data[pitch + bytes] = 9; // a texel not sampled
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R16G16B16A16_FLOAT), (1, 1, 4));
        // read as 4-byte texels the same bytes land elsewhere
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (0, 0, 4));
        assert_eq!(texel_bytes(DXGI_FORMAT_R11G11B10_FLOAT), 4);
        assert_eq!(texel_bytes(DXGI_FORMAT_R8_UNORM), 1);
        for f in [DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_B5G6R5_UNORM, DXGI_FORMAT_B5G5R5A1_UNORM, DXGI_FORMAT_B4G4R4A4_UNORM] {
            assert_eq!(texel_bytes(f), 2, "{f:?}");
        }
    }

    #[test]
    fn a_cleared_target_is_not_zero_but_has_no_texel_that_differs() {
        // 8 x 8 of R8G8B8A8 cleared to (0, 0, 0, 1): every texel not zero, none differs
        let (w, h, pitch) = (8usize, 8usize, 32usize);
        let mut data: Vec<u8> = [0, 0, 0, 255].repeat(w * h);
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (4, 0, 4));
        // one draw over texel (4, 4)
        data[4 * pitch + 4 * 4] = 200;
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_R8G8B8A8_UNORM), (4, 1, 4));
    }

    #[test]
    fn depth_formats_have_their_texel_sizes() {
        for f in [DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24G8_TYPELESS] {
            assert_eq!(texel_bytes(f), 4, "{f:?}");
        }
        assert_eq!(texel_bytes(DXGI_FORMAT_D32_FLOAT_S8X24_UINT), 8);
        assert_eq!(texel_bytes(DXGI_FORMAT_R32G8X24_TYPELESS), 8);
        assert_eq!(texel_bytes(DXGI_FORMAT_D16_UNORM), 2);
    }

    /// One present's target dump reads back 1 GiB at most: the targets past it, in the order
    /// dumped, are left out; a first target larger than that still goes.
    #[test]
    fn a_target_dump_stops_at_a_gib_of_readback() {
        let desc = |w: u64, h: u32, f: DXGI_FORMAT| D3D12_RESOURCE_DESC { Width: w, Height: h, Format: f, ..Default::default() };
        let hdr = readback_bytes(&desc(2560, 1440, DXGI_FORMAT_R32G32B32A32_FLOAT));
        assert_eq!(hdr, 2560 * 1440 * 16);
        // 64 of them: 3.8 GB; 18 fit in 1 GiB
        assert_eq!(dumped_within(&[hdr; 64], DUMP_BYTES_MOST), 18);
        // to_dump asks once per target, with the sizes so far: a target cut stays cut
        for n in 1..=64 {
            assert_eq!(dumped_within(&vec![hdr; n], DUMP_BYTES_MOST), n.min(18), "{n} targets");
        }
        let small = readback_bytes(&desc(1920, 1080, DXGI_FORMAT_R8_UNORM));
        assert_eq!(small, 1920 * 1080);
        assert_eq!(dumped_within(&[small; 64], DUMP_BYTES_MOST), 64, "under the cap: all");
        assert_eq!(dumped_within(&[], DUMP_BYTES_MOST), 0);
        // the order stays: a small target after the one past the cap is left out too
        let half = readback_bytes(&desc(16384, 16384, DXGI_FORMAT_R8G8_UNORM));
        assert_eq!(half, 512 << 20);
        assert_eq!(dumped_within(&[half, half, small], DUMP_BYTES_MOST), 2, "exactly 1 GiB");
        assert_eq!(dumped_within(&[half, half + 1, small], DUMP_BYTES_MOST), 1);
        // a first target larger than the cap goes alone
        let huge = readback_bytes(&desc(16384, 16384, DXGI_FORMAT_R32G32B32A32_FLOAT));
        assert!(huge > DUMP_BYTES_MOST);
        assert_eq!(dumped_within(&[huge], DUMP_BYTES_MOST), 1);
        assert_eq!(dumped_within(&[huge, small], DUMP_BYTES_MOST), 1);
    }

    /// A written target lists its pipelines by draws (12 at most), each pipeline's inputs noted
    /// at its first draw of the frame; the blend line names an inverting logic op.
    #[test]
    fn a_target_lists_its_pipelines_by_draws() {
        let mut f = Frame::default();
        assert!(f.drew_with(&[0x10], 0xa));
        assert!(!f.drew_with(&[0x10, 0x20], 0xa));
        assert!(f.drew_with(&[0x10], 0xb));
        for p in 0x100..0x110 {
            f.drew_with(&[0x30], p);
        }
        assert_eq!(f.pipelines_of(0x10, 12), (vec![(0xa, 2), (0xb, 1)], 0));
        assert_eq!(f.pipelines_of(0x20, 12), (vec![(0xa, 1)], 0));
        let (v, more) = f.pipelines_of(0x30, 12);
        assert_eq!((v.len(), v[0], more), (12, (0x100, 1), 4));
        let mut b = D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() };
        assert_eq!(blend_text(&b), "blend off, write mask 0xf");
        b.BlendEnable = true.into();
        b.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        b.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        b.BlendOp = D3D12_BLEND_OP_ADD;
        b.SrcBlendAlpha = D3D12_BLEND_ONE;
        b.DestBlendAlpha = D3D12_BLEND_ZERO;
        b.BlendOpAlpha = D3D12_BLEND_OP_REV_SUBTRACT;
        b.LogicOpEnable = true.into();
        b.LogicOp = D3D12_LOGIC_OP_COPY_INVERTED;
        assert_eq!(blend_text(&b), "blend on: colour SRC_ALPHA ADD INV_SRC_ALPHA, alpha ONE REV_SUBTRACT ZERO, logic op COPY_INVERTED, write mask 0xf");
        assert_eq!(shader_stem(17, "ps", 0xab), "target17_ps_00000000000000ab");
    }

    /// Compute pipelines are listed in order of first use, each with its direct and indirect
    /// dispatches, the first direct one's groups, the largest, and the first indirect arguments.
    #[test]
    fn compute_pipelines_list_in_order_of_first_use() {
        let mut f = Frame::default();
        f.dispatched(0xb, Dispatched::Indirect(None));
        f.dispatched(0xa, Dispatched::Direct([80, 45, 1]));
        f.dispatched(0xb, Dispatched::Indirect(Some("arguments 1x2x3 (max 1)".into())));
        f.dispatched(0xa, Dispatched::Direct([160, 90, 1]));
        f.dispatched(0xa, Dispatched::Direct([1, 1, 1]));
        f.dispatched(0xb, Dispatched::Direct([4, 4, 4]));
        let order: Vec<usize> = f.computes.iter().map(|c| c.pso).collect();
        assert_eq!(order, vec![0xb, 0xa]);
        let b = &f.computes[0];
        assert_eq!((b.direct, b.indirect, b.first, b.largest), (1, 2, Some([4, 4, 4]), Some([4, 4, 4])));
        assert_eq!(b.first_indirect, Some(None), "the first indirect one's arguments were on the GPU");
        let a = &f.computes[1];
        assert_eq!((a.direct, a.indirect, a.first, a.largest, a.first_indirect.clone()), (3, 0, Some([80, 45, 1]), Some([160, 90, 1]), None));
        assert_eq!(f.work, 0, "the work is counted by note_dispatch");
        assert_eq!(groups_text(a.first), "80x45x1");
        assert_eq!(groups_text(None), "none");
        assert_eq!(compute_stem(399, 0xab), "compute399_cs_00000000000000ab");
    }

    #[test]
    fn the_census_stops_at_the_end_of_the_buffer() {
        // 8 x 8 of D32S8X24 (8 bytes), a 256-byte pitch, but the buffer ends in the last row
        // (the footprint's last row is not padded to its pitch): its texels are not read
        let (w, h, pitch) = (8usize, 8usize, 256usize);
        let mut data = vec![1u8; pitch * 4 + 4 * 8 + 4];
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_D32_FLOAT_S8X24_UINT), (3, 0, 3));
        data.extend_from_slice(&[0; 4]);
        assert_eq!(census(&data, w, h, pitch, DXGI_FORMAT_D32_FLOAT_S8X24_UINT), (4, 1, 4));
    }
}
