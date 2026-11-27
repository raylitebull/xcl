//! `XCL_CAPTURE_FRAME=<work>[,<after_s>]`: one frame's command stream, written to disk, so a
//! rendering fault can be studied without running the title again for every question.
//!
//! The frame is the first with at least `work` draws and dispatches, not before `after_s` seconds
//! after the first present, and not the first frame recorded (its lists began before the
//! recording did). It goes to `<logs>/capture_<frame>/`:
//! - `frame.jsonl`: one object per operation in the order the GPU got them – the submits in their
//!   order, each list's operations in the order the title recorded them – with `i` (its place)
//!   and, for a draw or dispatch, `n` (its number among them), the render targets bound, the root
//!   arguments (only the ones changed since the list's last draw or dispatch where that is
//!   smaller: `"delta": true`) and how it went;
//! - `resources.jsonl`, `pipelines.jsonl`: what the operations name;
//! - `shaders/`: each referenced pipeline's console shaders translated again and disassembled;
//! - `summary.txt`: counts by kind, pass and pipeline, and what looks wrong.
//!
//! What is recorded is what the title called (on its lists: `title_list`), as it called it; a
//! barrier also as XCL sent it where that differs. XCL's own commands inside a title's call are
//! not recorded, and nothing recorded changes what goes to the PC. Lists are recorded by the
//! object the title holds and submits (a list split into segments is one recording). With the
//! switch on, every frame is recorded until the one written; the writing is done on a thread of
//! its own after that frame's present. Unset, each entry point checks one cached value.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::c_void;
use std::io::Write;
use std::mem::transmute_copy;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::sharded::Sharded;

/// operations of a frame written at most (and of one list's recording kept at most)
const OPS_MOST: usize = 2_000_000;
/// argument bytes of one ExecuteIndirect read at most
const INDIRECT_BYTES_MOST: usize = 16 << 10;
/// descriptor slots at the start of each root table whose resources a draw or dispatch names
pub(super) const TABLE_SLOTS: usize = 16;
/// passes listed in `summary.txt` at most
const PASSES_LISTED: usize = 400;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    /// draws and dispatches the frame needs
    work: u64,
    /// none before this long after the first present
    after: Duration,
}

/// `<work>[,<after_s>]`
fn parse(v: &str) -> Option<Config> {
    let mut it = v.split(',').map(str::trim);
    let work = it.next()?.parse::<u64>().ok().filter(|&w| w > 0)?;
    let after = match it.next() {
        Some(s) => Duration::try_from_secs_f64(s.parse::<f64>().ok()?).ok()?,
        None => Duration::ZERO,
    };
    it.next().is_none().then_some(Config { work, after })
}

fn config() -> Option<Config> {
    static C: OnceLock<Option<Config>> = OnceLock::new();
    *C.get_or_init(|| {
        let v = std::env::var("XCL_CAPTURE_FRAME").ok()?;
        let c = parse(&v);
        if c.is_none() {
            crate::xlog_warn!("D3D12.X: XCL_CAPTURE_FRAME={v:?} is not <draws and dispatches>[,<seconds after the first present>]: no frame captured");
        }
        c
    })
}

const UNSET: u8 = 0;
const OFF: u8 = 1;
/// the switch is on, the recording has not begun
const WAITING: u8 = 2;
const RECORDING: u8 = 3;
/// the frame was taken (or the switch is off)
const DONE: u8 = 4;
static STATE: AtomicU8 = AtomicU8::new(UNSET);

fn state() -> u8 {
    let s = STATE.load(Ordering::Relaxed);
    if s != UNSET {
        return s;
    }
    let _ = STATE.compare_exchange(UNSET, if config().is_some() { WAITING } else { OFF }, Ordering::Relaxed, Ordering::Relaxed);
    STATE.load(Ordering::Relaxed)
}

/// The switch is on and no frame was taken yet: what is made is noted (resources, views,
/// pipelines, command signatures), for whichever frame is captured.
pub fn enabled() -> bool {
    matches!(state(), WAITING | RECORDING)
}

/// Operations are being recorded.
#[inline]
pub fn on() -> bool {
    state() == RECORDING
}

// ---------------------------------------------------------------------------------------
// what an operation holds
// ---------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(super) enum RootVal {
    /// a descriptor table: the 32-bit console address of its first descriptor
    Table(u32),
    Constants(Vec<u32>),
    /// a root CBV / SRV / UAV: its GPU address
    View(u64),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct RootArg {
    pub i: u32,
    pub v: RootVal,
}

/// A render target or depth-stencil view: its handle, resource, format and subresource.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct View {
    handle: usize,
    res: usize,
    format: i32,
    mip: u32,
    slice: u32,
}

/// Where a copy reads or writes.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Loc {
    None,
    /// a buffer at an offset
    Buffer(usize, u64),
    /// a subresource of a texture
    Sub(usize, u32),
    /// a placed footprint in a buffer: offset, format, width, height, depth, row pitch
    Footprint(usize, u64, i32, u32, u32, u32, u32),
    /// a GPU address (CopyMemoryX)
    Address(u64),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum WorkKind {
    Draw { vertices: u32, instances: u32, first_vertex: u32, first_instance: u32 },
    DrawIndexed { indices: u32, instances: u32, first_index: u32, base_vertex: i32, first_instance: u32 },
    /// DrawIndexedX: indices read from an address
    DrawIndexedX { va: u64, format: i32, indices: u32 },
    Dispatch([u32; 3]),
    /// DispatchX with thread dimensions (flag 0x20): the thread counts, the flags and the groups
    /// they launch (None: the pipeline's group shape was not known when it was recorded)
    DispatchThreads { threads: [u32; 3], flags: u32, groups: Option<[u32; 3]> },
    DispatchMesh([u32; 3]),
    Indirect(Box<Indirect>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Indirect {
    sig: usize,
    /// the signature's stride and argument types (0 / none: made before the switch)
    stride: u32,
    types: Vec<i32>,
    dispatches: bool,
    args: usize,
    args_off: u64,
    count: usize,
    count_off: u64,
    max: u32,
    /// the arguments and the count as the CPU saw them when the call was recorded (None: not in
    /// CPU-readable title memory)
    values: Option<Vec<u32>>,
    count_value: Option<u32>,
}

/// How a draw or dispatch went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// on its translated pipeline
    Ran,
    Skipped,
    /// with a PC pipeline (untranslated)
    Pc,
    /// with the stand-in of a pipeline still being built: drew nothing
    StandIn,
    /// put off to the submit (its pipeline was being built)
    Deferred,
}

impl Outcome {
    fn name(self) -> &'static str {
        match self {
            Outcome::Ran => "ran",
            Outcome::Skipped => "skipped",
            Outcome::Pc => "pc",
            Outcome::StandIn => "standin",
            Outcome::Deferred => "deferred",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Work {
    kind: WorkKind,
    /// the title's pipeline
    pso: usize,
    /// graphics root arguments (else compute)
    gfx: bool,
    targets: Vec<usize>,
    dsv: Option<usize>,
    /// the root arguments: all of them, or (`delta`) the ones changed since the list's last
    /// draw or dispatch with the same kind of arguments
    root: Vec<RootArg>,
    delta: bool,
    heaps: [u32; 2],
    /// resources the first slots of its descriptor tables name
    views: Vec<usize>,
    outcome: Option<Outcome>,
}

impl Work {
    fn dispatch(&self) -> bool {
        match &self.kind {
            WorkKind::Dispatch(_) | WorkKind::DispatchThreads { .. } => true,
            WorkKind::Indirect(i) => i.dispatches,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Op {
    /// a list's recording begins (with its initial pipeline)
    Begin(usize),
    End,
    /// a list goes on after a KickoffX submitted what it held
    Continue,
    Pipeline(usize),
    RootSig { gfx: bool, rs: usize },
    /// the heaps set: (heap, its console address, sampler)
    Heaps(Vec<(usize, u32, bool)>),
    Targets { rtvs: Vec<View>, dsv: Option<View> },
    Viewports(Vec<[f32; 6]>),
    Scissors(Vec<[i32; 4]>),
    /// as the title gave them, as XCL sent them, and how many depth plane copies went around them
    Barrier { title: Vec<[u32; 8]>, sent: Vec<[u32; 8]>, plane_copies: usize },
    Work(Box<Work>),
    Copy { kind: &'static str, dst: Loc, src: Loc, at: [u32; 3], bx: Option<[u32; 6]>, bytes: Option<u64>, refused: bool },
    /// rtv / dsv / uav_uint / uav_float: the view (handle), its resource, the values (floats as
    /// their bits), the rectangles, whether it was not done
    Clear { kind: &'static str, view: u64, res: usize, values: [u32; 4], flags: u32, rects: u32, refused: bool },
    Resolve { dst: usize, dst_sub: u32, src: usize, src_sub: u32, format: i32, region: Option<(u32, u32, Option<[i32; 4]>, i32)> },
}

impl Op {
    fn name(&self) -> &'static str {
        match self {
            Op::Begin(_) => "list_begin",
            Op::End => "list_end",
            Op::Continue => "list_continue",
            Op::Pipeline(_) => "set_pipeline",
            Op::RootSig { .. } => "root_signature",
            Op::Heaps(_) => "heaps",
            Op::Targets { .. } => "set_render_targets",
            Op::Viewports(_) => "viewport",
            Op::Scissors(_) => "scissor",
            Op::Barrier { .. } => "barrier",
            Op::Work(w) => match w.kind {
                WorkKind::Draw { .. } => "draw",
                WorkKind::DrawIndexed { .. } => "draw_indexed",
                WorkKind::DrawIndexedX { .. } => "draw_indexed_x",
                WorkKind::Dispatch(_) | WorkKind::DispatchThreads { .. } => "dispatch",
                WorkKind::DispatchMesh(_) => "dispatch_mesh",
                WorkKind::Indirect(_) => "execute_indirect",
            },
            Op::Copy { .. } => "copy",
            Op::Clear { .. } => "clear",
            Op::Resolve { .. } => "resolve",
        }
    }
}

/// The root arguments to record of a draw or dispatch, given the list's previous ones of that
/// kind: (only the changed ones, them) where that is fewer, else (false, all).
pub(super) fn root_delta(prev: Option<&[RootArg]>, now: &[RootArg]) -> (bool, Vec<RootArg>) {
    match prev {
        Some(p) if p.len() == now.len() => {
            let changed: Vec<RootArg> = now.iter().filter(|a| !p.contains(a)).cloned().collect();
            if changed.len() < now.len() {
                (true, changed)
            } else {
                (false, now.to_vec())
            }
        }
        _ => (false, now.to_vec()),
    }
}

/// The heap and slot a root table at console address `va` starts at, from the heaps set (view,
/// sampler): the nearest base below it.
pub(super) fn table_index(va: u32, heaps: [u32; 2]) -> Option<(&'static str, u32)> {
    [("view", heaps[0], super::console_heap::SLOT), ("sampler", heaps[1], super::console_heap::SAMPLER_SLOT)]
        .into_iter()
        .filter(|&(_, base, _)| base != 0 && va >= base)
        .min_by_key(|&(_, base, _)| va - base)
        .map(|(heap, base, slot)| (heap, (va - base) / slot))
}

/// A view's mip and first slice from the 3 words of its description's union (`dsv`: a
/// depth-stencil view's dimensions).
pub(super) fn mip_slice(dsv: bool, dim: i32, w: [u32; 3]) -> (u32, u32) {
    match (dsv, dim) {
        // RTV: 1D, 2D; DSV: 1D, 2D
        (false, 2 | 4) | (true, 1 | 3) => (w[0], 0),
        // RTV: 1D array, 2D array, 3D (first W slice); DSV: 1D array, 2D array
        (false, 3 | 5 | 8) | (true, 2 | 4) => (w[0], w[1]),
        // multisampled arrays
        (false, 7) | (true, 6) => (0, w[0]),
        _ => (0, 0),
    }
}

// ---------------------------------------------------------------------------------------
// recordings
// ---------------------------------------------------------------------------------------

/// What a list recorded since its Reset.
#[derive(Default)]
struct Rec {
    ops: Vec<Op>,
    work: u64,
    /// the recording as closed (Close), with its draws and dispatches
    closed: Option<(Arc<Vec<Op>>, u64)>,
    seen: bool,
    /// the list began before the recording did
    partial: bool,
    /// operations left out past `OPS_MOST`
    dropped: u64,
    /// the root arguments of the last draw or dispatch: compute, graphics
    prev: [Option<Vec<RootArg>>; 2],
}

/// `op` onto `r`: its place there (None: past the bound)
fn push_to(r: &mut Rec, mut op: Op, most: usize) -> Option<usize> {
    if !r.seen {
        r.seen = true;
        r.partial = !matches!(op, Op::Begin(_));
    }
    if r.ops.len() >= most {
        r.dropped += 1;
        return None;
    }
    match &mut op {
        Op::RootSig { gfx, .. } => r.prev[*gfx as usize] = None,
        Op::Work(w) => {
            let b = w.gfx as usize;
            let full = std::mem::take(&mut w.root);
            (w.delta, w.root) = root_delta(r.prev[b].as_deref(), &full);
            r.prev[b] = Some(full);
            r.work += 1;
        }
        _ => {}
    }
    r.ops.push(op);
    Some(r.ops.len() - 1)
}

fn begin(r: &mut Rec, pso: usize) {
    *r = Rec::default();
    push_to(r, Op::Begin(pso), OPS_MOST);
}

fn close(r: &mut Rec) {
    if !r.seen {
        return;
    }
    push_to(r, Op::End, OPS_MOST + 1);
    r.closed = Some((Arc::new(std::mem::take(&mut r.ops)), std::mem::take(&mut r.work)));
}

/// the recording of each list, by the object the title has
static RECS: Sharded<Rec> = Sharded::new();

thread_local! {
    /// entry points of the title's this thread is in (the ones inside the first are XCL's)
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// the operation the outermost of them recorded (list, place)
    static CURRENT: Cell<Option<(usize, usize)>> = const { Cell::new(None) };
}

/// An entry point of the title's being recorded (`op`), until it returns.
pub(super) struct Guard {
    entered: bool,
    outer: bool,
    at: Option<(usize, usize)>,
}

impl Guard {
    const NONE: Guard = Guard { entered: false, outer: false, at: None };

    /// Changes the operation recorded (what XCL sent, whether it was done).
    pub(super) fn amend(&self, f: impl FnOnce(&mut Op)) {
        if let Some((key, i)) = self.at {
            RECS.with(key, |r| {
                if let Some(op) = r.ops.get_mut(i) {
                    f(op)
                }
            });
        }
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.entered {
            DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            if self.outer {
                CURRENT.with(|c| c.set(None));
            }
        }
    }
}

/// The object a title's call on the PC list `this` was made on (a split list's segments are one).
fn key_of(this: *mut c_void) -> usize {
    super::title_list::wrapper_of(this).map_or(this as usize, |w| w as usize)
}

/// The title called an entry point on `this`: `make` is what it did (None: nothing to record).
/// Calls inside it (XCL's) are not recorded, nor the calls a submit records for put-off
/// dispatches (they were recorded when the title made them).
pub(super) fn op(this: *mut c_void, make: impl FnOnce() -> Option<Op>) -> Guard {
    if !on() {
        return Guard::NONE;
    }
    let depth = DEPTH.with(|d| d.replace(d.get() + 1));
    if depth > 0 || super::pso_async::is_at_submit() {
        return Guard { entered: true, outer: depth == 0, at: None };
    }
    let key = key_of(this);
    let at = make().and_then(|op| RECS.with(key, |r| push_to(r, op, OPS_MOST))).map(|i| (key, i));
    CURRENT.with(|c| c.set(at));
    Guard { entered: true, outer: true, at }
}

/// How the draw or dispatch being recorded on this thread went (the first word counts).
pub(super) fn outcome(o: Outcome) {
    if !on() {
        return;
    }
    let Some((key, i)) = CURRENT.with(|c| c.get()) else { return };
    let pso = RECS.with(key, |r| match r.ops.get(i) {
        Some(Op::Work(w)) if w.outcome.is_none() => Some(w.pso),
        _ => None,
    });
    let Some(pso) = pso else { return };
    // (not drawn, or with a PC pipeline, while its real one is being built: its stand-in's)
    let o = if matches!(o, Outcome::Skipped | Outcome::Pc) && super::pso_async::is_pending(pso) { Outcome::StandIn } else { o };
    RECS.with(key, |r| {
        if let Some(Op::Work(w)) = r.ops.get_mut(i) {
            w.outcome.get_or_insert(o);
        }
    });
}

/// A list `key` (the title's object) begins a recording with pipeline `pso` (made, or Reset).
pub(super) fn list_begin(key: usize, pso: usize) {
    if on() {
        RECS.with(key, |r| begin(r, pso));
    }
}

/// The title closed list `key`.
pub(super) fn list_end(key: usize) {
    if on() {
        RECS.with(key, close);
    }
}

/// List `key` is gone.
pub(super) fn forget(key: usize) {
    if on() {
        RECS.remove(key);
    }
}

/// A draw or dispatch on `this` with the title's pipeline `pso`: the render targets bound now and
/// the root arguments (`root`, of the graphics ones with `gfx`) with the heaps set.
pub(super) unsafe fn work(this: *mut c_void, kind: WorkKind, pso: usize, gfx: bool, root: Vec<RootArg>, heaps: [u32; 2], views: Vec<usize>) -> Op {
    let mut targets: Vec<usize> = super::list_x::bound_render_targets(this).into_iter().filter_map(view_res).collect();
    targets.dedup();
    let dsv = super::list_x::bound_dsv(this).and_then(view_res);
    for &r in targets.iter().chain(dsv.iter()).chain(views.iter()) {
        describe(r);
    }
    Op::Work(Box::new(Work { kind, pso, gfx, targets, dsv, root, delta: false, heaps, views, outcome: None }))
}

/// An ExecuteIndirect's call: with its arguments and count as the CPU sees them now, where they
/// are in title memory it can read.
pub(super) unsafe fn indirect(sig: usize, dispatches: bool, max: u32, args: usize, args_off: u64, count: usize, count_off: u64) -> WorkKind {
    let (stride, types) = SIGNATURES.lock().unwrap().as_ref().and_then(|m| m.get(&sig).cloned()).unwrap_or((0, Vec::new()));
    describe(args);
    describe(count);
    let values = super::real::console_buffer(args).and_then(|(p, n)| {
        let bytes = (max as usize).saturating_mul(stride as usize).min(INDIRECT_BYTES_MOST).min(n.saturating_sub(args_off as usize));
        let at = p + args_off as usize;
        (stride != 0 && bytes >= 4 && crate::gpu::shader::readable(at, bytes)).then(|| (0..bytes / 4).map(|k| ((at + 4 * k) as *const u32).read_unaligned()).collect())
    });
    let count_value = super::real::console_buffer(count)
        .filter(|&(_, n)| count_off as usize + 4 <= n)
        .map(|(p, _)| p + count_off as usize)
        .filter(|&at| crate::gpu::shader::readable(at, 4))
        .map(|at| (at as *const u32).read_unaligned());
    WorkKind::Indirect(Box::new(Indirect { sig, stride, types, dispatches, args, args_off, count, count_off, max, values, count_value }))
}

/// The render targets `this` has bound now (after an OMSetRenderTargets).
pub(super) unsafe fn targets_op(this: *mut c_void) -> Op {
    let rtvs = super::list_x::bound_render_targets(this).into_iter().map(|h| view(h)).collect();
    let dsv = super::list_x::bound_dsv(this).map(|h| view(h));
    Op::Targets { rtvs, dsv }
}

/// A copy location (a texture's subresource, or a footprint in a buffer).
pub(super) unsafe fn location(l: *const D3D12_TEXTURE_COPY_LOCATION) -> Loc {
    let Some(l) = l.as_ref() else { return Loc::None };
    let r: usize = transmute_copy(&l.pResource);
    describe(r);
    if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
        let f = l.Anonymous.PlacedFootprint;
        Loc::Footprint(r, f.Offset, f.Footprint.Format.0, f.Footprint.Width, f.Footprint.Height, f.Footprint.Depth, f.Footprint.RowPitch)
    } else {
        Loc::Sub(r, l.Anonymous.SubresourceIndex)
    }
}

/// A buffer location (described).
pub(super) unsafe fn buffer(res: usize, offset: u64) -> Loc {
    describe(res);
    Loc::Buffer(res, offset)
}

/// A clear through `view` (an RTV / DSV handle, or a UAV's GPU handle with its resource `res`).
pub(super) unsafe fn clear(kind: &'static str, view: u64, res: Option<usize>, values: [u32; 4], flags: u32, rects: u32) -> Op {
    let res = res.or_else(|| view_res(view as usize)).unwrap_or(0);
    describe(res);
    Op::Clear { kind, view, res, values, flags, rects, refused: false }
}

/// The resources of a barrier call (described), and the call.
pub(super) unsafe fn barrier(title: &[[u32; 8]]) -> Op {
    for b in title {
        for r in barrier_resources(b) {
            describe(r);
        }
    }
    Op::Barrier { title: title.to_vec(), sent: title.to_vec(), plane_copies: 0 }
}

fn barrier_resources(b: &[u32; 8]) -> Vec<usize> {
    let r = |lo: usize| (b[lo] as u64 | (b[lo + 1] as u64) << 32) as usize;
    match b[0] {
        0 | 2 => vec![r(2)],
        1 => vec![r(2), r(4)],
        _ => vec![],
    }
    .into_iter()
    .filter(|&r| r != 0)
    .collect()
}

/// A resource some operation names: described for the frame (once a frame).
pub(super) unsafe fn describe(r: usize) -> usize {
    if r == 0 || !on() {
        return r;
    }
    let e = EPOCH.load(Ordering::Relaxed);
    if DESCS.with(r, |i| i.epoch == e) {
        return r;
    }
    let p = r as *mut c_void;
    let desc = ID3D12Resource::from_raw_borrowed(&p).map(|x| x.GetDesc());
    let title = super::real::title_address(r);
    let kept_at = super::title_textures::address(r);
    let uma_buffer = super::real::console_buffer(r).is_some();
    DESCS.with(r, |i| *i = ResInfo { epoch: e, title, kept_at, uma_buffer, desc });
    r
}

/// The resource of an RTV / DSV handle.
fn view_res(handle: usize) -> Option<usize> {
    VIEWS.get(handle).map(|v| v.res).filter(|&r| r != 0).or_else(|| super::present::rtv_resource(handle))
}

unsafe fn view(handle: usize) -> View {
    let mut v = VIEWS.get(handle).unwrap_or_default();
    if v.res == 0 {
        v.res = super::present::rtv_resource(handle).unwrap_or(0);
    }
    v.handle = handle;
    describe(v.res);
    v
}

// ---------------------------------------------------------------------------------------
// what was made (kept while the switch is on, whichever frame is captured)
// ---------------------------------------------------------------------------------------

#[derive(Clone, Default)]
struct ResInfo {
    /// the frame it was described in
    epoch: u64,
    title: Option<usize>,
    /// a texture kept in step with the title's memory there (`title_textures`)
    kept_at: Option<usize>,
    /// a console buffer in unified memory
    uma_buffer: bool,
    desc: Option<D3D12_RESOURCE_DESC>,
}

#[derive(Clone, Default)]
struct Created {
    how: String,
    /// the console description's format, flags and layout
    console: Option<[u32; 3]>,
    va: Option<usize>,
    kept: bool,
    /// its width and height then (a resource made later at its address through another path is not it)
    size: (u64, u32),
}

/// resources as the frames described them
static DESCS: Sharded<ResInfo> = Sharded::new();
static CREATED: Sharded<Created> = Sharded::new();
/// RTV / DSV handles: what their views name
static VIEWS: Sharded<View> = Sharded::new();
/// command signatures: stride, argument types
static SIGNATURES: Mutex<Option<HashMap<usize, (u32, Vec<i32>)>>> = Mutex::new(None);
/// frames recorded (resources are described once each)
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// Resource `res` was made by `how` (`lut3d::created`'s arguments).
pub(super) unsafe fn created(how: &str, res: *mut c_void, console: Option<&[u8; 0x38]>, va: Option<usize>, kept: bool) {
    if !enabled() || res.is_null() {
        return;
    }
    let u = |x: &[u8; 0x38], o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let size = ID3D12Resource::from_raw_borrowed(&res).map(|r| r.GetDesc()).map_or((0, 0), |d| (d.Width, d.Height));
    CREATED.insert(res as usize, Created { how: how.to_string(), console: console.map(|x| [u(x, 0x20), u(x, 0x30), u(x, 0x2C)]), va, kept, size });
}

/// A render target view (`dsv`: a depth-stencil view) of `res` made at `handle`; `desc`: the six
/// words of its description as the PC got it (format, dimension, then the union – a DSV's after
/// its flags).
pub(super) fn note_view(handle: usize, res: usize, dsv: bool, desc: Option<[u32; 6]>) {
    if !enabled() {
        return;
    }
    let v = match desc {
        Some(w) => {
            let u = if dsv { [w[3], w[4], w[5]] } else { [w[2], w[3], w[4]] };
            let (mip, slice) = mip_slice(dsv, w[1] as i32, u);
            View { handle, res, format: w[0] as i32, mip, slice }
        }
        None => View { handle, res, ..Default::default() },
    };
    VIEWS.insert(handle, v);
}

/// A command signature: its stride and argument types (PC ones).
pub(super) fn note_signature(sig: usize, stride: u32, types: &[i32]) {
    if enabled() {
        SIGNATURES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, (stride, types.to_vec()));
    }
}

/// A pipeline as the title made it.
struct Pipe {
    kind: &'static str,
    /// `real::shaders_key` of its shaders (graphics)
    set: Option<u64>,
    /// (stage, `real::shader_key`, dump hash, bytes kept)
    stages: Vec<(&'static str, u64, u64, Option<Arc<[u8]>>)>,
    state: Value,
    split: bool,
}

static PIPES: Mutex<Option<HashMap<usize, Arc<Pipe>>>> = Mutex::new(None);

fn keep_pipe(pso: usize, p: Pipe) {
    PIPES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, Arc::new(p));
}

fn stages_of(stages: &[(&'static str, &[u8])]) -> Vec<(&'static str, u64, u64, Option<Arc<[u8]>>)> {
    stages
        .iter()
        .map(|&(s, b)| {
            let (key, hash, blob) = super::targets::keep_shader(b);
            (s, key, hash, blob)
        })
        .collect()
}

/// Graphics pipeline `pso` was made from `d` with `stages` (VS, PS, DS, HS, GS as present).
pub(super) fn note_graphics(pso: usize, d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, stages: &[(&'static str, &[u8])], split: bool) {
    if !enabled() || pso == 0 {
        return;
    }
    let set = super::real::shaders_key(&stages.iter().map(|(_, b)| b.to_vec()).collect::<Vec<_>>());
    keep_pipe(pso, Pipe { kind: "graphics", set: Some(set), stages: stages_of(stages), state: graphics_state(d), split });
}

/// Compute pipeline `pso` was made from `cs`.
pub(super) fn note_compute(pso: usize, cs: &[u8]) {
    if !enabled() || pso == 0 {
        return;
    }
    keep_pipe(pso, Pipe { kind: "compute", set: None, stages: stages_of(&[("cs", cs)]), state: Value::Null, split: false });
}

/// Pipeline `pso` was made from a pipeline stream (one the graphics or compute path did not note).
pub(super) fn note_stream(pso: usize) {
    if !enabled() || pso == 0 {
        return;
    }
    let mut g = PIPES.lock().unwrap();
    g.get_or_insert_with(HashMap::new).entry(pso).or_insert_with(|| Arc::new(Pipe { kind: "stream", set: None, stages: Vec::new(), state: Value::Null, split: false }));
}

fn graphics_state(d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> Value {
    let n = (d.NumRenderTargets as usize).min(8);
    let b = &d.BlendState;
    let rts = if b.IndependentBlendEnable.as_bool() { n.max(1) } else { 1 };
    let blend: Vec<Value> = b.RenderTarget[..rts].iter().map(|r| Value::String(super::targets::blend_text(r))).collect();
    let r = &d.RasterizerState;
    let s = &d.DepthStencilState;
    let face = |f: &D3D12_DEPTH_STENCILOP_DESC| json!({"fail": f.StencilFailOp.0, "depth_fail": f.StencilDepthFailOp.0, "pass": f.StencilPassOp.0, "func": f.StencilFunc.0});
    json!({
        "blend": {"alpha_to_coverage": b.AlphaToCoverageEnable.as_bool(), "independent": b.IndependentBlendEnable.as_bool(), "targets": blend},
        "sample_mask": hex(d.SampleMask as u64),
        "raster": {"fill": r.FillMode.0, "cull": r.CullMode.0, "front_ccw": r.FrontCounterClockwise.as_bool(), "depth_bias": r.DepthBias, "depth_bias_clamp": r.DepthBiasClamp, "slope_scaled_depth_bias": r.SlopeScaledDepthBias, "depth_clip": r.DepthClipEnable.as_bool(), "conservative": r.ConservativeRaster.0},
        "depth_stencil": {"depth": s.DepthEnable.as_bool(), "write_mask": s.DepthWriteMask.0, "func": s.DepthFunc.0, "stencil": s.StencilEnable.as_bool(), "read_mask": s.StencilReadMask, "stencil_write_mask": s.StencilWriteMask, "front": face(&s.FrontFace), "back": face(&s.BackFace)},
        "rt_formats": d.RTVFormats[..n].iter().map(|f| f.0).collect::<Vec<_>>(),
        "ds_format": d.DSVFormat.0,
        "topology_type": d.PrimitiveTopologyType.0,
        "samples": d.SampleDesc.Count,
        "strip_cut": d.IBStripCutValue.0,
    })
}

// ---------------------------------------------------------------------------------------
// the frame
// ---------------------------------------------------------------------------------------

/// Whether a submitted list had a recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Recorded {
    Yes,
    /// it began before the recording did: its first operations are missing
    Partial,
    /// none (XCL's own, or recorded before the capture and not reset since)
    No,
}

pub(super) enum Item {
    Submit { queue: usize, lists: Vec<(usize, Recorded)>, kickoff: bool },
    List { key: usize, ops: Arc<Vec<Op>> },
    Present { queue: usize, res: Vec<usize>, frame: u64 },
}

#[derive(Default)]
pub(super) struct Frame {
    items: Vec<Item>,
    work: u64,
    ops: usize,
    /// operations left out: past the frame's bound, and past a list's
    truncated: u64,
    dropped: u64,
    /// the first frame recorded (its lists began before)
    first: bool,
}

/// A submitted list's recording: (its operations, draws and dispatches, partial, operations
/// it left out); None: no recording.
type Part = Option<(Arc<Vec<Op>>, u64, bool, u64)>;

/// Of `n` more operations onto `total`, how many fit in `most`.
pub(super) fn admit(total: usize, n: usize, most: usize) -> usize {
    n.min(most.saturating_sub(total))
}

impl Frame {
    /// A submit to `queue` of the lists `parts` (in their order).
    fn submit(&mut self, queue: usize, parts: Vec<(usize, Part)>, kickoff: bool, most: usize) {
        let lists = parts
            .iter()
            .map(|(k, p)| (*k, p.as_ref().map_or(Recorded::No, |p| if p.2 { Recorded::Partial } else { Recorded::Yes })))
            .collect();
        self.items.push(Item::Submit { queue, lists, kickoff });
        for (key, part) in parts {
            let Some((ops, work, _, dropped)) = part else { continue };
            let take = admit(self.ops, ops.len(), most);
            self.truncated += (ops.len() - take) as u64;
            self.dropped += dropped;
            if take == 0 {
                continue;
            }
            let whole = take == ops.len();
            self.work += if whole { work } else { ops[..take].iter().filter(|o| matches!(o, Op::Work(_))).count() as u64 };
            let ops = if whole { ops } else { Arc::new(ops[..take].to_vec()) };
            self.ops += take;
            self.items.push(Item::List { key, ops });
        }
    }
}

static FRAME: Mutex<Option<Frame>> = Mutex::new(None);

/// The title submits `lists` (its objects) to `queue`.
pub(super) unsafe fn submit(queue: *mut c_void, lists: &[*mut c_void]) {
    if !on() {
        return;
    }
    let parts: Vec<(usize, Part)> = lists
        .iter()
        .map(|&l| {
            let key = l as usize;
            let part = RECS.with(key, |r| {
                r.seen.then(|| match &r.closed {
                    Some((ops, work)) => (ops.clone(), *work, r.partial, r.dropped),
                    None => (Arc::new(r.ops.clone()), r.work, r.partial, r.dropped),
                })
            });
            (key, part)
        })
        .collect();
    if let Some(f) = FRAME.lock().unwrap().as_mut() {
        f.submit(queue as usize, parts, false, OPS_MOST);
    }
}

/// KickoffX on `this` to `queue`: what the list recorded so far is submitted, and it goes on.
pub(super) fn kickoff(this: *mut c_void, queue: usize) {
    if !on() {
        return;
    }
    let key = key_of(this);
    let part = RECS.with(key, |r| {
        let part = r.seen.then(|| (Arc::new(std::mem::take(&mut r.ops)), std::mem::take(&mut r.work), r.partial, r.dropped));
        // (what follows is the same recording; its root arguments start over)
        r.prev = [None, None];
        if r.seen {
            push_to(r, Op::Continue, OPS_MOST);
        }
        part
    });
    if let Some(f) = FRAME.lock().unwrap().as_mut() {
        f.submit(queue, vec![(key, part)], true, OPS_MOST);
    }
}

/// Whether frame `f` is the one to write.
fn due(c: &Config, f: &Frame) -> bool {
    !f.first && f.work >= c.work
}

/// The title presents `res` on `queue` (the `frame`th present): the frame ends.
pub(super) unsafe fn present(queue: usize, res: &[usize], frame: u64) {
    let s = state();
    if s != WAITING && s != RECORDING {
        return;
    }
    let Some(c) = config() else { return };
    static T0: OnceLock<Instant> = OnceLock::new();
    let t0 = *T0.get_or_init(Instant::now);
    if s == WAITING {
        if t0.elapsed() >= c.after {
            *FRAME.lock().unwrap() = Some(Frame { first: true, ..Default::default() });
            STATE.store(RECORDING, Ordering::Relaxed);
            crate::xlog_info!("D3D12.X: XCL_CAPTURE_FRAME: recording from present {frame} on, until a frame has {} draws and dispatches", c.work);
        }
        return;
    }
    for &r in res {
        describe(r);
    }
    let mut f = FRAME.lock().unwrap().take().unwrap_or_default();
    f.items.push(Item::Present { queue, res: res.to_vec(), frame });
    if !due(&c, &f) {
        EPOCH.fetch_add(1, Ordering::Relaxed);
        *FRAME.lock().unwrap() = Some(Frame::default());
        return;
    }
    STATE.store(DONE, Ordering::Relaxed);
    let mut keys = Vec::new();
    RECS.for_each(|k, _| keys.push(k));
    for k in keys {
        RECS.remove(k);
    }
    let Some(logs) = crate::state::boot().map(|b| b.logs_dir()) else { return };
    let dir = logs.join(format!("capture_{frame}"));
    crate::xlog_info!("D3D12.X: XCL_CAPTURE_FRAME: frame {frame} captured ({} draws and dispatches, {} operations): writing {}", f.work, f.ops, dir.display());
    let _ = std::thread::Builder::new().name("xcl-capture".into()).spawn(move || {
        match write(&dir, &f, frame) {
            Ok(()) => crate::xlog_info!("D3D12.X: XCL_CAPTURE_FRAME: {} written", dir.display()),
            Err(e) => crate::xlog_warn!("D3D12.X: XCL_CAPTURE_FRAME: writing {}: {e}", dir.display()),
        }
    });
}

// ---------------------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------------------

/// An operation of the frame in execution order.
pub(super) enum Entry<'a> {
    Op { list: usize, op: &'a Op, n: Option<u64> },
    Item(&'a Item),
}

/// The frame's operations in the order the GPU got them, draws and dispatches numbered.
pub(super) fn flatten(items: &[Item]) -> Vec<Entry<'_>> {
    let mut out = Vec::new();
    let mut n = 0u64;
    for it in items {
        match it {
            Item::List { key, ops } => {
                for op in ops.iter() {
                    let k = matches!(op, Op::Work(_)).then(|| {
                        n += 1;
                        n - 1
                    });
                    out.push(Entry::Op { list: *key, op, n: k });
                }
            }
            _ => out.push(Entry::Item(it)),
        }
    }
    out
}

fn hex(v: u64) -> String {
    format!("{v:#x}")
}

fn res_json(r: usize) -> Value {
    if r == 0 {
        return Value::Null;
    }
    match DESCS.get(r).and_then(|i| i.title.or(i.kept_at)) {
        Some(t) => json!({"pc": hex(r as u64), "title": hex(t as u64)}),
        None => json!({"pc": hex(r as u64)}),
    }
}

fn sub_json(s: u32) -> Value {
    if s == u32::MAX {
        json!("all")
    } else {
        json!(s)
    }
}

fn view_json(v: &View) -> Value {
    let d = DESCS.get(v.res).and_then(|i| i.desc);
    json!({
        "handle": hex(v.handle as u64),
        "res": res_json(v.res),
        "w": d.map(|d| d.Width),
        "h": d.map(|d| d.Height),
        "format": if v.format != 0 { v.format } else { d.map_or(0, |d| d.Format.0) },
        "mip": v.mip,
        "slice": v.slice,
    })
}

fn loc_json(l: &Loc) -> Value {
    match *l {
        Loc::None => Value::Null,
        Loc::Buffer(r, o) => json!({"res": res_json(r), "offset": o}),
        Loc::Sub(r, s) => json!({"res": res_json(r), "sub": sub_json(s)}),
        Loc::Footprint(r, o, f, w, h, d, p) => json!({"res": res_json(r), "footprint": {"offset": o, "format": f, "w": w, "h": h, "d": d, "row_pitch": p}}),
        Loc::Address(va) => json!({"address": hex(va)}),
    }
}

fn barrier_json(b: &[u32; 8]) -> Value {
    let r = |lo: usize| (b[lo] as u64 | (b[lo + 1] as u64) << 32) as usize;
    match b[0] {
        0 => json!({"type": "transition", "res": res_json(r(2)), "sub": sub_json(b[4]), "before": hex(b[5] as u64), "after": hex(b[6] as u64), "flags": b[1]}),
        1 => json!({"type": "aliasing", "before": res_json(r(2)), "after": res_json(r(4)), "flags": b[1]}),
        2 => json!({"type": "uav", "res": res_json(r(2)), "flags": b[1]}),
        t => json!({"type": t, "words": b.iter().map(|&w| hex(w as u64)).collect::<Vec<_>>()}),
    }
}

fn root_json(a: &RootArg, heaps: [u32; 2]) -> Value {
    match &a.v {
        RootVal::Table(va) => {
            let (heap, index) = table_index(*va, heaps).map_or((None, None), |(h, k)| (Some(h), Some(k)));
            json!({"i": a.i, "table": {"va": hex(*va as u64), "heap": heap, "index": index}})
        }
        RootVal::Constants(w) => json!({"i": a.i, "constants": w.iter().map(|&w| format!("{w:#010x}")).collect::<Vec<_>>()}),
        RootVal::View(va) => json!({"i": a.i, "view": hex(*va)}),
    }
}

/// A pipeline's kind and shaders, as the frame's lines name it.
fn pipe_brief(pso: usize) -> Map<String, Value> {
    let mut m = Map::new();
    let p = PIPES.lock().unwrap().as_ref().and_then(|m| m.get(&pso).cloned());
    m.insert("kind".into(), json!(pipe_kind(pso, p.as_deref())));
    if let Some(p) = p {
        m.insert("shaders".into(), Value::Object(p.stages.iter().map(|(s, k, _, _)| (s.to_string(), json!(format!("{k:016x}")))).collect()));
        m.insert("hashes".into(), Value::Object(p.stages.iter().map(|(s, _, h, _)| (s.to_string(), json!(format!("{h:016x}")))).collect()));
    }
    m
}

fn pipe_kind(pso: usize, p: Option<&Pipe>) -> String {
    let real = super::pso_async::peek(pso);
    if let Some(tp) = super::real::two_pass_of(real) {
        return format!("mesh two-pass ({})", tp.kind);
    }
    if let Some(p) = p {
        return p.kind.to_string();
    }
    if super::real::is_translated_compute(real) || super::real::is_console_compute(pso) {
        "compute".into()
    } else if super::real::is_translated_graphics(real) {
        "graphics".into()
    } else {
        "unknown (made before the switch, or not a console pipeline)".into()
    }
}

fn work_json(w: &Work, m: &mut Map<String, Value>) {
    let mut put = |k: &str, v: Value| {
        m.insert(k.into(), v);
    };
    match &w.kind {
        WorkKind::Draw { vertices, instances, first_vertex, first_instance } => {
            put("vertices", json!(vertices));
            put("instances", json!(instances));
            put("first_vertex", json!(first_vertex));
            put("first_instance", json!(first_instance));
        }
        WorkKind::DrawIndexed { indices, instances, first_index, base_vertex, first_instance } => {
            put("indices", json!(indices));
            put("instances", json!(instances));
            put("first_index", json!(first_index));
            put("base_vertex", json!(base_vertex));
            put("first_instance", json!(first_instance));
        }
        WorkKind::DrawIndexedX { va, format, indices } => {
            put("index_address", json!(hex(*va)));
            put("index_format", json!(format));
            put("indices", json!(indices));
        }
        WorkKind::Dispatch(g) | WorkKind::DispatchMesh(g) => put("groups", json!(g)),
        WorkKind::DispatchThreads { threads, flags, groups } => {
            put("groups", json!(groups));
            put("threads", json!(threads));
            put("dispatch_flags", json!(hex(*flags as u64)));
        }
        WorkKind::Indirect(i) => {
            put("signature", json!(hex(i.sig as u64)));
            put("stride", json!(i.stride));
            put("argument_types", json!(i.types));
            put("dispatches", json!(i.dispatches));
            put("arguments", json!({"res": res_json(i.args), "offset": i.args_off}));
            put("count", if i.count == 0 { Value::Null } else { json!({"res": res_json(i.count), "offset": i.count_off}) });
            put("max_count", json!(i.max));
            put("values", i.values.as_ref().map_or(Value::Null, |v| json!(v)));
            put("count_value", json!(i.count_value));
        }
    }
    put("pso", json!(hex(w.pso as u64)));
    put("targets", Value::Array(w.targets.iter().map(|&r| res_json(r)).collect()));
    put("dsv", w.dsv.map_or(Value::Null, res_json));
    put("root", json!({"bind": if w.gfx { "graphics" } else { "compute" }, "delta": w.delta, "args": w.root.iter().map(|a| root_json(a, w.heaps)).collect::<Vec<_>>()}));
    put("views", Value::Array(w.views.iter().map(|&r| res_json(r)).collect()));
    put("outcome", json!(w.outcome.map_or("unknown", Outcome::name)));
}

fn entry_json(i: usize, e: &Entry) -> Value {
    let mut m = Map::new();
    m.insert("i".into(), json!(i));
    match e {
        Entry::Item(Item::Submit { queue, lists, kickoff }) => {
            m.insert("op".into(), json!("submit"));
            m.insert("queue".into(), json!(hex(*queue as u64)));
            m.insert(
                "lists".into(),
                Value::Array(lists.iter().map(|(l, r)| json!({"list": hex(*l as u64), "recorded": match r { Recorded::Yes => json!(true), Recorded::Partial => json!("partial"), Recorded::No => json!(false) }})).collect()),
            );
            if *kickoff {
                m.insert("kickoff".into(), json!(true));
            }
        }
        Entry::Item(Item::Present { queue, res, frame }) => {
            m.insert("op".into(), json!("present"));
            m.insert("queue".into(), json!(hex(*queue as u64)));
            m.insert("resources".into(), Value::Array(res.iter().map(|&r| res_json(r)).collect()));
            m.insert("frame".into(), json!(frame));
        }
        Entry::Item(Item::List { .. }) => {}
        Entry::Op { list, op, n } => {
            m.insert("op".into(), json!(op.name()));
            m.insert("list".into(), json!(hex(*list as u64)));
            if let Some(n) = n {
                m.insert("n".into(), json!(n));
            }
            match op {
                Op::Begin(pso) => {
                    m.insert("pso".into(), if *pso == 0 { Value::Null } else { json!(hex(*pso as u64)) });
                }
                Op::End | Op::Continue => {}
                Op::Pipeline(pso) => {
                    m.insert("pso".into(), json!(hex(*pso as u64)));
                    m.extend(pipe_brief(*pso));
                }
                Op::RootSig { gfx, rs } => {
                    m.insert("bind".into(), json!(if *gfx { "graphics" } else { "compute" }));
                    m.insert("rs".into(), json!(hex(*rs as u64)));
                }
                Op::Heaps(h) => {
                    m.insert("heaps".into(), Value::Array(h.iter().map(|&(o, base, s)| json!({"heap": hex(o as u64), "address": hex(base as u64), "sampler": s})).collect()));
                }
                Op::Targets { rtvs, dsv } => {
                    m.insert("rtvs".into(), Value::Array(rtvs.iter().map(view_json).collect()));
                    m.insert("dsv".into(), dsv.as_ref().map_or(Value::Null, view_json));
                }
                Op::Viewports(v) => {
                    m.insert("viewports".into(), json!(v));
                }
                Op::Scissors(r) => {
                    m.insert("rects".into(), json!(r));
                }
                Op::Barrier { title, sent, plane_copies } => {
                    m.insert("barriers".into(), Value::Array(title.iter().map(barrier_json).collect()));
                    if title != sent {
                        m.insert("sent".into(), Value::Array(sent.iter().map(barrier_json).collect()));
                        m.insert("rewritten".into(), json!(true));
                    }
                    if *plane_copies > 0 {
                        m.insert("plane_copies".into(), json!(plane_copies));
                    }
                }
                Op::Work(w) => work_json(w, &mut m),
                Op::Copy { kind, dst, src, at, bx, bytes, refused } => {
                    m.insert("kind".into(), json!(kind));
                    m.insert("dst".into(), loc_json(dst));
                    m.insert("src".into(), loc_json(src));
                    if *at != [0; 3] {
                        m.insert("at".into(), json!(at));
                    }
                    if let Some(b) = bx {
                        m.insert("box".into(), json!(b));
                    }
                    if let Some(b) = bytes {
                        m.insert("bytes".into(), json!(b));
                    }
                    if *refused {
                        m.insert("refused".into(), json!(true));
                    }
                }
                Op::Clear { kind, view, res, values, flags, rects, refused } => {
                    m.insert("kind".into(), json!(kind));
                    m.insert("view".into(), json!(hex(*view)));
                    m.insert("res".into(), res_json(*res));
                    let v = match *kind {
                        "uav_uint" => json!(values),
                        "dsv" => json!({"depth": f32::from_bits(values[0]), "stencil": values[1], "flags": flags}),
                        _ => json!(values.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>()),
                    };
                    m.insert("values".into(), v);
                    m.insert("rects".into(), json!(rects));
                    if *refused {
                        m.insert("refused".into(), json!(true));
                    }
                }
                Op::Resolve { dst, dst_sub, src, src_sub, format, region } => {
                    m.insert("dst".into(), json!({"res": res_json(*dst), "sub": sub_json(*dst_sub)}));
                    m.insert("src".into(), json!({"res": res_json(*src), "sub": sub_json(*src_sub)}));
                    m.insert("format".into(), json!(format));
                    if let Some((x, y, rect, mode)) = region {
                        m.insert("region".into(), json!({"x": x, "y": y, "rect": rect, "mode": mode}));
                    }
                }
            }
        }
    }
    Value::Object(m)
}

/// The resources an operation names.
fn resources_of(op: &Op, out: &mut BTreeSet<usize>) {
    let mut add = |r: usize| {
        if r != 0 {
            out.insert(r);
        }
    };
    let loc = |l: &Loc| match *l {
        Loc::Buffer(r, _) | Loc::Sub(r, _) | Loc::Footprint(r, ..) => r,
        _ => 0,
    };
    match op {
        Op::Targets { rtvs, dsv } => rtvs.iter().chain(dsv.iter()).for_each(|v| add(v.res)),
        Op::Barrier { title, sent, .. } => title.iter().chain(sent.iter()).flat_map(barrier_resources).for_each(add),
        Op::Work(w) => {
            w.targets.iter().chain(w.dsv.iter()).chain(w.views.iter()).for_each(|&r| add(r));
            if let WorkKind::Indirect(i) = &w.kind {
                add(i.args);
                add(i.count);
            }
        }
        Op::Copy { dst, src, .. } => {
            add(loc(dst));
            add(loc(src));
        }
        Op::Clear { res, .. } => add(*res),
        Op::Resolve { dst, src, .. } => {
            add(*dst);
            add(*src);
        }
        _ => {}
    }
}

fn resource_json(r: usize) -> Value {
    let i = DESCS.get(r).unwrap_or_default();
    let d = i.desc;
    let c = CREATED.get(r).filter(|c| d.is_some_and(|d| (d.Width, d.Height) == c.size));
    let dimension = d.map(|d| match d.Dimension {
        D3D12_RESOURCE_DIMENSION_BUFFER => "buffer",
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => "texture1d",
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => "texture2d",
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => "texture3d",
        _ => "unknown",
    });
    let (how, kept) = match &c {
        Some(c) => (Some(c.how.clone()), Some(c.kept)),
        None if i.uma_buffer => (Some("a console buffer in unified memory (the title's memory itself)".to_string()), Some(true)),
        None if i.kept_at.is_some() => (Some("a texture kept in step with the title's memory (uploaded from it)".to_string()), Some(true)),
        None => (None, None),
    };
    json!({
        "pc": hex(r as u64),
        "title": i.title.or(i.kept_at).or(c.as_ref().and_then(|c| c.va)).map(|a| hex(a as u64)),
        "dimension": dimension,
        "width": d.map(|d| d.Width),
        "height": d.map(|d| d.Height),
        "depth_or_array": d.map(|d| d.DepthOrArraySize),
        "mips": d.map(|d| d.MipLevels),
        "format": d.map(|d| d.Format.0),
        "console_format": c.as_ref().and_then(|c| c.console).map(|x| x[0]),
        "console_flags": c.as_ref().and_then(|c| c.console).map(|x| hex(x[1] as u64)),
        "console_layout": c.as_ref().and_then(|c| c.console).map(|x| hex(x[2] as u64)),
        "flags": d.map(|d| hex(d.Flags.0 as u64)),
        "samples": d.map(|d| d.SampleDesc.Count),
        "layout": d.map(|d| d.Layout.0),
        "created": how,
        "kept_in_step": kept,
    })
}

/// What the frame's draws and dispatches with one pipeline did.
#[derive(Default)]
struct PipeUse {
    draws: u64,
    dispatches: u64,
    outcomes: BTreeMap<&'static str, u64>,
}

/// A pass: draws and dispatches in a row with the same render targets.
#[derive(Debug, PartialEq)]
pub(super) struct Pass {
    pub targets: Vec<usize>,
    pub dsv: Option<usize>,
    pub draws: u64,
    pub dispatches: u64,
    pub pipelines: BTreeSet<usize>,
    pub first: usize,
    pub last: usize,
}

/// The passes of the frame's operations (`flatten`'s, by their place).
pub(super) fn passes(entries: &[Entry]) -> Vec<Pass> {
    let mut out: Vec<Pass> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let Entry::Op { op: Op::Work(w), .. } = e else { continue };
        let same = out.last().is_some_and(|p| p.targets == w.targets && p.dsv == w.dsv);
        if !same {
            out.push(Pass { targets: w.targets.clone(), dsv: w.dsv, draws: 0, dispatches: 0, pipelines: BTreeSet::new(), first: i, last: i });
        }
        let p = out.last_mut().unwrap();
        if w.dispatch() {
            p.dispatches += 1;
        } else {
            p.draws += 1;
        }
        p.pipelines.insert(w.pso);
        p.last = i;
    }
    out
}

/// groups one ExecuteIndirect may run past which the guard drops a dispatch (`indirect_guard`'s
/// default `XCL_INDIRECT_MAX_GROUPS`)
const GUARD_GROUPS: u64 = 4 << 20;

fn write(dir: &std::path::Path, f: &Frame, frame: u64) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.join("shaders"))?;
    let entries = flatten(&f.items);
    let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut resources = BTreeSet::new();
    let mut psos = BTreeSet::new();
    let mut uses: BTreeMap<usize, PipeUse> = BTreeMap::new();
    let mut warn: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let mut note = |what: &'static str, at: String| warn.entry(what).or_default().push(at);
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("frame.jsonl"))?);
        for (i, e) in entries.iter().enumerate() {
            match e {
                Entry::Op { op, n, .. } => {
                    *counts.entry(op.name()).or_default() += 1;
                    resources_of(op, &mut resources);
                    match op {
                        Op::Pipeline(p) => {
                            psos.insert(*p);
                        }
                        Op::Work(w) => {
                            psos.insert(w.pso);
                            let u = uses.entry(w.pso).or_default();
                            if w.dispatch() {
                                u.dispatches += 1;
                            } else {
                                u.draws += 1;
                            }
                            let o = w.outcome.map_or("unknown", Outcome::name);
                            *u.outcomes.entry(o).or_default() += 1;
                            let at = format!("n={} i={i}", n.unwrap_or(0));
                            if w.outcome == Some(Outcome::StandIn) {
                                note("draws or dispatches with a stand-in pipeline (drew nothing)", at.clone());
                            }
                            if w.outcome == Some(Outcome::Skipped) {
                                note("draws or dispatches XCL skipped", at.clone());
                            }
                            match &w.kind {
                                WorkKind::Dispatch(g) | WorkKind::DispatchMesh(g) | WorkKind::DispatchThreads { threads: g, .. } if g.contains(&0) => note("dispatches with zero groups", at.clone()),
                                WorkKind::Indirect(x) => {
                                    if x.max == 0 {
                                        note("ExecuteIndirects with a max count of 0", at.clone());
                                    }
                                    if x.dispatches && x.stride >= 12 {
                                        let big = x.values.as_ref().is_some_and(|v| {
                                            let per = x.stride as usize / 4;
                                            let n = x.count_value.map_or(x.max, |c| c.min(x.max)) as usize;
                                            // (a plain dispatch signature: its groups are its last 3 words)
                                            v.chunks(per).take(n).any(|c| c.len() >= 3 && c[c.len() - 3..].iter().map(|&g| g as u64).product::<u64>() > GUARD_GROUPS)
                                        });
                                        if big {
                                            note("ExecuteIndirect dispatches past the guard's group count (the guard runs zeros there)", at.clone());
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        Op::Copy { refused: true, .. } | Op::Clear { refused: true, .. } => note("copies or clears XCL did not do", format!("i={i}")),
                        Op::Barrier { title, sent, .. } if title != sent => note("barrier calls XCL rewrote (state bits dropped, transitions dropped, UAV barriers widened or added, depth twins)", format!("i={i}")),
                        _ => {}
                    }
                }
                Entry::Item(Item::Submit { lists, .. }) => {
                    *counts.entry("submit").or_default() += 1;
                    for (l, r) in lists {
                        match r {
                            Recorded::Partial => note("lists that began before the recording (their first operations are missing)", format!("i={i} list {l:#x}")),
                            Recorded::No => note("lists submitted with no recording (XCL's own, or recorded before the capture and not reset since)", format!("i={i} list {l:#x}")),
                            Recorded::Yes => {}
                        }
                    }
                }
                Entry::Item(Item::Present { res, .. }) => {
                    *counts.entry("present").or_default() += 1;
                    resources.extend(res.iter().copied().filter(|&r| r != 0));
                }
                Entry::Item(_) => {}
            }
            serde_json::to_writer(&mut out, &entry_json(i, e))?;
            out.write_all(b"\n")?;
        }
        out.flush()?;
    }
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("resources.jsonl"))?);
        for &r in &resources {
            serde_json::to_writer(&mut out, &resource_json(r))?;
            out.write_all(b"\n")?;
        }
        out.flush()?;
    }
    let pipes = PIPES.lock().unwrap().clone().unwrap_or_default();
    let mut shaders: Vec<super::targets::ShaderFile> = Vec::new();
    let mut written = BTreeSet::new();
    {
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join("pipelines.jsonl"))?);
        for &pso in &psos {
            let p = pipes.get(&pso);
            let real = super::pso_async::peek(pso);
            let u = uses.get(&pso);
            let mut m = pipe_brief(pso);
            m.insert("pso".into(), json!(hex(pso as u64)));
            m.insert("set".into(), json!(p.and_then(|p| p.set).map(|s| format!("{s:016x}"))));
            m.insert("state".into(), p.map_or(Value::Null, |p| p.state.clone()));
            m.insert("wave32".into(), json!(p.map(|p| p.split)));
            m.insert("translated_now".into(), json!(super::real::is_translated_graphics(real) || super::real::is_translated_compute(real) || super::real::two_pass_of(real).is_some()));
            m.insert("building_now".into(), json!(super::pso_async::is_pending(pso)));
            m.insert("draws".into(), json!(u.map_or(0, |u| u.draws)));
            m.insert("dispatches".into(), json!(u.map_or(0, |u| u.dispatches)));
            m.insert("outcomes".into(), json!(u.map(|u| u.outcomes.clone())));
            serde_json::to_writer(&mut out, &Value::Object(m))?;
            out.write_all(b"\n")?;
            if let Some(p) = p {
                let vs = p.stages.iter().find(|s| s.0 == "vs").and_then(|s| s.3.clone());
                for (s, key, _, blob) in &p.stages {
                    if let Some(b) = blob.as_ref().filter(|_| written.insert((*s, *key))) {
                        shaders.push((format!("shaders/{s}_{key:016x}"), s, b.clone(), vs.clone(), p.split));
                    }
                }
            }
        }
        out.flush()?;
    }
    write_summary(dir, f, frame, &entries, &counts, &uses, &pipes, &warn)?;
    let n = shaders.len();
    super::targets::write_shaders(dir, shaders, super::targets::HLSL_BYTES_MOST);
    crate::xlog_info!("D3D12.X: XCL_CAPTURE_FRAME: {n} shaders translated again");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_summary(
    dir: &std::path::Path,
    f: &Frame,
    frame: u64,
    entries: &[Entry],
    counts: &BTreeMap<&'static str, u64>,
    uses: &BTreeMap<usize, PipeUse>,
    pipes: &HashMap<usize, Arc<Pipe>>,
    warn: &BTreeMap<&'static str, Vec<String>>,
) -> std::io::Result<()> {
    let mut s = String::new();
    use std::fmt::Write as _;
    let _ = writeln!(s, "XCL_CAPTURE_FRAME: frame {frame}: {} draws and dispatches, {} operations ({} lines)", f.work, f.ops, entries.len());
    if f.truncated > 0 {
        let _ = writeln!(s, "TRUNCATED: {} operations past the bound of {OPS_MOST} left out", f.truncated);
    }
    if f.dropped > 0 {
        let _ = writeln!(s, "TRUNCATED: {} operations past one list's bound of {OPS_MOST} were not recorded", f.dropped);
    }
    let _ = writeln!(s, "\noperations by kind:");
    for (k, n) in counts {
        let _ = writeln!(s, "  {k:<20} {n}");
    }
    let ps = passes(entries);
    let short = |r: &usize| format!("{r:#x}");
    let _ = writeln!(s, "\npasses (draws and dispatches in a row with the same render targets): {}", ps.len());
    for (k, p) in ps.iter().enumerate().take(PASSES_LISTED) {
        let t: Vec<String> = p.targets.iter().map(short).collect();
        let _ = writeln!(
            s,
            "  pass {k}: i {}..{}, targets [{}] dsv {}: {} draws, {} dispatches, {} pipelines",
            p.first,
            p.last,
            t.join(" "),
            p.dsv.as_ref().map_or("-".into(), short),
            p.draws,
            p.dispatches,
            p.pipelines.len()
        );
    }
    if ps.len() > PASSES_LISTED {
        let _ = writeln!(s, "  and {} more", ps.len() - PASSES_LISTED);
    }
    let _ = writeln!(s, "\npipelines by draws and dispatches: {}", uses.len());
    let mut by: Vec<(&usize, &PipeUse)> = uses.iter().collect();
    by.sort_by(|a, b| (b.1.draws + b.1.dispatches).cmp(&(a.1.draws + a.1.dispatches)).then(a.0.cmp(b.0)));
    for (pso, u) in by {
        let p = pipes.get(pso);
        let shaders: Vec<String> = p.map_or(Vec::new(), |p| p.stages.iter().map(|(s, k, _, _)| format!("{s} {k:016x}")).collect());
        let o: Vec<String> = u.outcomes.iter().map(|(k, n)| format!("{k} {n}")).collect();
        let _ = writeln!(s, "  {pso:#x} {}: {} draws, {} dispatches ({}) {}", pipe_kind(*pso, p.map(|p| &**p)), u.draws, u.dispatches, o.join(", "), shaders.join(" "));
    }
    let _ = writeln!(s, "\nwarnings:");
    if warn.is_empty() {
        let _ = writeln!(s, "  none");
    }
    for (what, at) in warn {
        let shown: Vec<&str> = at.iter().take(24).map(|x| x.as_str()).collect();
        let _ = writeln!(s, "  {} {what}: {}{}", at.len(), shown.join(", "), if at.len() > 24 { ", …" } else { "" });
    }
    let _ = writeln!(
        s,
        "\nnot recorded: console-shaped lists (XCL_CONSOLE_LISTS), bundles' contents, ExecuteBundle, enhanced barriers, descriptor copies and \
         writes, CopyTiles, queries, predication, vertex and index buffers, topology, blend factor, stencil reference, depth bounds, \
         ClearState, the memory writes, waits and cache flushes of the console's list methods (#60-#72, #88-#90, #131, #132, #146)"
    );
    std::fs::write(dir.join("summary.txt"), s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(pso: usize, targets: &[usize], root: Vec<RootArg>) -> Op {
        Op::Work(Box::new(Work {
            kind: WorkKind::Draw { vertices: 3, instances: 1, first_vertex: 0, first_instance: 0 },
            pso,
            gfx: true,
            targets: targets.to_vec(),
            dsv: None,
            root,
            delta: false,
            heaps: [0x1000, 0x8000],
            views: Vec::new(),
            outcome: None,
        }))
    }

    fn dispatch(pso: usize, root: Vec<RootArg>) -> Op {
        Op::Work(Box::new(Work {
            kind: WorkKind::Dispatch([4, 4, 1]),
            pso,
            gfx: false,
            targets: Vec::new(),
            dsv: None,
            root,
            delta: false,
            heaps: [0; 2],
            views: Vec::new(),
            outcome: None,
        }))
    }

    fn c(i: u32, w: u32) -> RootArg {
        RootArg { i, v: RootVal::Constants(vec![w]) }
    }

    fn part(r: &Rec) -> Part {
        r.closed.as_ref().map(|(ops, w)| (ops.clone(), *w, r.partial, r.dropped))
    }

    /// A `DispatchX` with thread dimensions is written as a dispatch with the groups it launched
    /// (null: not known when recorded), its thread counts and its flags.
    #[test]
    fn a_dispatch_with_thread_dimensions_is_written_with_its_threads_and_groups() {
        let Op::Work(mut w) = dispatch(0xd0, vec![]) else { unreachable!() };
        w.kind = WorkKind::DispatchThreads { threads: [100, 1, 1], flags: 0x8020, groups: Some([2, 1, 1]) };
        assert!(w.dispatch());
        assert_eq!(Op::Work(w.clone()).name(), "dispatch");
        let mut m = Map::new();
        work_json(&w, &mut m);
        assert_eq!((&m["groups"], &m["threads"], &m["dispatch_flags"]), (&json!([2, 1, 1]), &json!([100, 1, 1]), &json!("0x8020")));
        w.kind = WorkKind::DispatchThreads { threads: [100, 1, 1], flags: 0x20, groups: None };
        let mut m = Map::new();
        work_json(&w, &mut m);
        assert_eq!(m["groups"], Value::Null);
    }

    #[test]
    fn the_switch_takes_a_work_threshold_and_seconds() {
        assert_eq!(parse("500"), Some(Config { work: 500, after: Duration::ZERO }));
        assert_eq!(parse(" 500 , 2.5 "), Some(Config { work: 500, after: Duration::from_millis(2500) }));
        for bad in ["0", "x", "500,x", "500,-1", "500,1,2", ""] {
            assert_eq!(parse(bad), None, "{bad:?}");
        }
        let c = Config { work: 10, after: Duration::ZERO };
        let f = |work, first| Frame { work, first, ..Default::default() };
        assert!(due(&c, &f(10, false)));
        assert!(!due(&c, &f(9, false)), "too little work");
        assert!(!due(&c, &f(100, true)), "the first frame recorded began before the recording");
    }

    /// Lists recorded at once (on several threads: their operations interleaved in time) come out
    /// in the order they were submitted, each in its own order, the draws numbered so.
    #[test]
    fn operations_come_out_in_the_order_they_were_executed() {
        let (mut a, mut b) = (Rec::default(), Rec::default());
        begin(&mut a, 0);
        begin(&mut b, 0);
        // a and b interleaved: a0 b0 a1 b1 b2
        push_to(&mut a, Op::Pipeline(0xa0), OPS_MOST);
        push_to(&mut b, Op::Pipeline(0xb0), OPS_MOST);
        push_to(&mut a, draw(0xa0, &[0x10], vec![]), OPS_MOST);
        push_to(&mut b, draw(0xb0, &[0x20], vec![]), OPS_MOST);
        push_to(&mut b, dispatch(0xb1, vec![]), OPS_MOST);
        close(&mut a);
        close(&mut b);
        // submitted b, then a (b on a second queue first), then a again
        let mut f = Frame::default();
        f.submit(0x51, vec![(0xb, part(&b)), (0xc, None)], false, OPS_MOST);
        f.submit(0x52, vec![(0xa, part(&a))], false, OPS_MOST);
        f.submit(0x52, vec![(0xa, part(&a))], false, OPS_MOST);
        assert_eq!(f.work, 2 + 1 + 1);
        let e = flatten(&f.items);
        let seq: Vec<String> = e
            .iter()
            .map(|e| match e {
                Entry::Op { list, op, n } => format!("{list:x}:{}{}", op.name(), n.map_or(String::new(), |n| format!("#{n}"))),
                Entry::Item(Item::Submit { queue, lists, .. }) => format!("submit {queue:x} {:?}", lists.iter().map(|l| l.1).collect::<Vec<_>>()),
                Entry::Item(_) => "?".into(),
            })
            .collect();
        assert_eq!(
            seq,
            [
                "submit 51 [Yes, No]",
                "b:list_begin",
                "b:set_pipeline",
                "b:draw#0",
                "b:dispatch#1",
                "b:list_end",
                "submit 52 [Yes]",
                "a:list_begin",
                "a:set_pipeline",
                "a:draw#2",
                "a:list_end",
                "submit 52 [Yes]",
                "a:list_begin",
                "a:set_pipeline",
                "a:draw#3",
                "a:list_end",
            ]
        );
        // the passes: b's draw into 0x20, b's dispatch (no targets bound), a's draws into 0x10
        let p = passes(&e);
        assert_eq!(p.iter().map(|p| (p.targets.clone(), p.draws, p.dispatches)).collect::<Vec<_>>(), vec![(vec![0x20], 1, 0), (vec![], 0, 1), (vec![0x10], 2, 0)]);
        assert_eq!((p[2].first, p[2].last), (9, 14));
        // a list recorded from the middle (no Begin): partial
        let mut late = Rec::default();
        push_to(&mut late, draw(1, &[], vec![]), OPS_MOST);
        close(&mut late);
        let mut f = Frame::default();
        f.submit(1, vec![(0xd, part(&late))], false, OPS_MOST);
        assert!(matches!(&f.items[0], Item::Submit { lists, .. } if lists[0].1 == Recorded::Partial));
    }

    /// Root arguments: all at a list's first draw and after a root signature is set, then only the
    /// changed ones where they are fewer – per kind (graphics, compute) – flagged as such.
    #[test]
    fn root_arguments_are_recorded_as_changed_since_the_last_draw() {
        assert_eq!(root_delta(None, &[c(0, 1), c(1, 2)]), (false, vec![c(0, 1), c(1, 2)]));
        assert_eq!(root_delta(Some(&[c(0, 1), c(1, 2)]), &[c(0, 1), c(1, 3)]), (true, vec![c(1, 3)]));
        assert_eq!(root_delta(Some(&[c(0, 1), c(1, 2)]), &[c(0, 1), c(1, 2)]), (true, vec![]), "nothing changed");
        assert_eq!(root_delta(Some(&[c(0, 1), c(1, 2)]), &[c(0, 5), c(1, 6)]), (false, vec![c(0, 5), c(1, 6)]), "all changed: not smaller");
        assert_eq!(root_delta(Some(&[c(0, 1)]), &[c(0, 1), c(1, 2)]), (false, vec![c(0, 1), c(1, 2)]), "another layout");
        let mut r = Rec::default();
        begin(&mut r, 0);
        let t = |i, va| RootArg { i, v: RootVal::Table(va) };
        let v = |i, va| RootArg { i, v: RootVal::View(va) };
        push_to(&mut r, draw(1, &[], vec![t(0, 0x100), c(1, 7), v(2, 0x9000)]), OPS_MOST);
        push_to(&mut r, dispatch(2, vec![c(0, 1), c(1, 2)]), OPS_MOST);
        push_to(&mut r, draw(1, &[], vec![t(0, 0x100), c(1, 8), v(2, 0x9000)]), OPS_MOST);
        push_to(&mut r, dispatch(2, vec![c(0, 1), c(1, 3)]), OPS_MOST);
        push_to(&mut r, Op::RootSig { gfx: true, rs: 9 }, OPS_MOST);
        push_to(&mut r, draw(1, &[], vec![t(0, 0x100), c(1, 8), v(2, 0x9000)]), OPS_MOST);
        let got: Vec<(bool, Vec<RootArg>)> = r.ops.iter().filter_map(|o| if let Op::Work(w) = o { Some((w.delta, w.root.clone())) } else { None }).collect();
        assert_eq!(
            got,
            vec![
                (false, vec![t(0, 0x100), c(1, 7), v(2, 0x9000)]),
                (false, vec![c(0, 1), c(1, 2)]),
                (true, vec![c(1, 8)]),
                (true, vec![c(1, 3)]),
                (false, vec![t(0, 0x100), c(1, 8), v(2, 0x9000)]),
            ]
        );
        // the heap slot a table starts at: the nearest heap below it
        assert_eq!(table_index(0x1000 + 5 * 32, [0x1000, 0x8000]), Some(("view", 5)));
        assert_eq!(table_index(0x8000 + 3 * 16, [0x1000, 0x8000]), Some(("sampler", 3)));
        assert_eq!(table_index(0x800, [0x1000, 0]), None);
        // a view's mip and first slice by its dimension (RTV 2D array, 3D; DSV 2D, MS array)
        assert_eq!(mip_slice(false, 5, [2, 7, 1]), (2, 7));
        assert_eq!(mip_slice(false, 8, [1, 4, 2]), (1, 4));
        assert_eq!(mip_slice(true, 3, [3, 9, 9]), (3, 0));
        assert_eq!(mip_slice(true, 6, [5, 1, 0]), (0, 5));
    }

    /// A frame of each kind of operation goes to `frame.jsonl` one object a line, in order, with the
    /// draws numbered and the root arguments as recorded; the other files are there.
    /// (`XCL_CAPTURE_TEST_DIR=<dir>`: written there and kept, for reading it with a script.)
    #[test]
    fn a_frame_is_written_one_operation_a_line() {
        let kept = std::env::var_os("XCL_CAPTURE_TEST_DIR").map(std::path::PathBuf::from);
        let dir = kept.clone().unwrap_or_else(|| std::env::temp_dir().join(format!("xcl-capture-test-{}", std::process::id())));
        let mut r = Rec::default();
        begin(&mut r, 0);
        let rt = View { handle: 0x500, res: 0x7000, format: 10, mip: 0, slice: 0 };
        for op in [
            Op::Heaps(vec![(0x60, 0x1000, false), (0x61, 0x8000, true)]),
            Op::Pipeline(0xa0),
            Op::RootSig { gfx: true, rs: 0x99 },
            Op::Targets { rtvs: vec![rt], dsv: None },
            Op::Viewports(vec![[0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0]]),
            Op::Scissors(vec![[0, 0, 1920, 1080]]),
            Op::Barrier { title: vec![[0, 0, 0x7000, 0, u32::MAX, 0x800_0000 | 4, 0x40, 0]], sent: vec![[0, 0, 0x7000, 0, u32::MAX, 4, 0x40, 0], [2, 0, 0, 0, 0, 0, 0, 0]], plane_copies: 0 },
            draw(0xa0, &[0x7000], vec![RootArg { i: 0, v: RootVal::Table(0x1000 + 64) }, c(1, 5)]),
            draw(0xa0, &[0x7000], vec![RootArg { i: 0, v: RootVal::Table(0x1000 + 64) }, c(1, 6)]),
            dispatch(0xb0, vec![c(0, 1)]),
            Op::Copy { kind: "texture_region", dst: Loc::Sub(0x7100, 0), src: Loc::Footprint(0x7200, 512, 28, 64, 64, 1, 256), at: [0; 3], bx: None, bytes: None, refused: false },
            Op::Clear { kind: "uav_float", view: 0x1040, res: 0x7100, values: [1.0f32.to_bits(), 0, 0, 0], flags: 0, rects: 0, refused: true },
            Op::Resolve { dst: 0x7300, dst_sub: 0, src: 0x7000, src_sub: 0, format: 28, region: None },
        ] {
            push_to(&mut r, op, OPS_MOST);
        }
        let mut ind = dispatch(0xb0, vec![c(0, 1)]);
        if let Op::Work(w) = &mut ind {
            w.kind = WorkKind::Indirect(Box::new(Indirect { sig: 0x40, stride: 12, types: vec![2], dispatches: true, args: 0x7400, args_off: 16, count: 0, count_off: 0, max: 2, values: Some(vec![8, 8, 1, 0, 1, 1]), count_value: None }));
            w.outcome = Some(Outcome::StandIn);
        }
        push_to(&mut r, ind, OPS_MOST);
        close(&mut r);
        let mut f = Frame::default();
        f.submit(0x51, vec![(0xa, part(&r)), (0xc, None)], false, OPS_MOST);
        f.items.push(Item::Present { queue: 0x51, res: vec![0x7300], frame: 7 });
        write(&dir, &f, 7).unwrap();
        let text = std::fs::read_to_string(dir.join("frame.jsonl")).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let ops: Vec<&str> = lines.iter().map(|l| l["op"].as_str().unwrap()).collect();
        assert_eq!(
            ops,
            [
                "submit", "list_begin", "heaps", "set_pipeline", "root_signature", "set_render_targets", "viewport", "scissor", "barrier", "draw", "draw", "dispatch", "copy", "clear",
                "resolve", "execute_indirect", "list_end", "present"
            ]
        );
        assert!(lines.iter().enumerate().all(|(i, l)| l["i"] == json!(i)));
        assert_eq!(lines[9]["n"], json!(0));
        assert_eq!(lines[9]["root"]["args"][0]["table"], json!({"va": "0x1040", "heap": "view", "index": 2}));
        assert_eq!((lines[10]["root"]["delta"].clone(), lines[10]["root"]["args"].clone()), (json!(true), json!([{"i": 1, "constants": ["0x00000006"]}])));
        assert_eq!(lines[10]["targets"], json!([{"pc": "0x7000"}]));
        assert_eq!(lines[8]["rewritten"], json!(true));
        assert_eq!(lines[8]["barriers"][0]["sub"], json!("all"));
        assert_eq!(lines[15]["values"], json!([8, 8, 1, 0, 1, 1]));
        assert_eq!(lines[15]["outcome"], json!("standin"));
        assert_eq!(lines[0]["lists"], json!([{"list": "0xa", "recorded": true}, {"list": "0xc", "recorded": false}]));
        let res = std::fs::read_to_string(dir.join("resources.jsonl")).unwrap();
        assert_eq!(res.lines().count(), 5, "0x7000 0x7100 0x7200 0x7300 0x7400: {res}");
        let pipes = std::fs::read_to_string(dir.join("pipelines.jsonl")).unwrap();
        assert_eq!(pipes.lines().count(), 2);
        let summary = std::fs::read_to_string(dir.join("summary.txt")).unwrap();
        assert!(summary.contains("passes (draws and dispatches in a row with the same render targets): 2"), "{summary}");
        assert!(summary.contains("1 barrier calls XCL rewrote"), "{summary}");
        assert!(summary.contains("1 copies or clears XCL did not do"), "{summary}");
        assert!(summary.contains("1 draws or dispatches with a stand-in pipeline"), "{summary}");
        if kept.is_none() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// A list keeps `most` operations (the rest counted as dropped), a frame takes as many as fit
    /// in its bound (the rest counted as truncated, and their draws not counted).
    #[test]
    fn a_frame_and_a_list_stop_at_their_bounds() {
        assert_eq!(admit(0, 5, 10), 5);
        assert_eq!(admit(8, 5, 10), 2);
        assert_eq!(admit(10, 5, 10), 0);
        assert_eq!(admit(12, 5, 10), 0);
        let mut r = Rec::default();
        begin(&mut r, 0);
        for _ in 0..6 {
            push_to(&mut r, draw(1, &[], vec![]), 4);
        }
        assert_eq!((r.ops.len(), r.dropped, r.work), (4, 3, 3), "begin + 3 draws kept, 3 dropped");
        close(&mut r);
        let ops = r.closed.as_ref().unwrap().0.clone();
        assert!(matches!(ops.last(), Some(Op::End)), "the end is kept past the bound");
        // a frame of 6 at most: the first list whole (5), one operation of the second
        let mut f = Frame::default();
        f.submit(1, vec![(0xa, part(&r)), (0xb, part(&r))], false, 6);
        assert_eq!((f.ops, f.truncated, f.dropped, f.work), (6, 4, 6, 3 + 0));
        f.submit(1, vec![(0xc, part(&r))], false, 6);
        assert_eq!((f.ops, f.truncated), (6, 9));
        assert_eq!(flatten(&f.items).iter().filter(|e| matches!(e, Entry::Op { .. })).count(), 6);
    }
}
