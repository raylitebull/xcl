//! What XCL does when a call into the graphics driver does not return or the driver reports an
//! error, and what it was doing when that began.
//!
//! A retail title's world ended twice in a driver error whose cause the log did not hold: the
//! process was killed from outside seconds after the driver's first event, before the device was
//! reported removed, and the second time it then hung half-terminated in the driver for five
//! minutes with the desktop behind it. Three things, all cheap enough to be always on:
//!
//! - a ring of the last draws and dispatches (pipeline, kind, size, frame), written when one is
//!   recorded on a list (an index and two stores) and read only for a report: which programs the
//!   title recorded last (not necessarily submitted, and a list submitted again is not in it twice);
//! - a thread (`xcl-gpu-watch`) that looks for a removed device ten times a second (a title
//!   whose threads all wait submits nothing and would never notice), and times the calls into
//!   the driver that return at once on a working system: an ExecuteCommandLists or a Present that
//!   has not returned after `XCL_GPU_STALL_MS` (2 000) is logged with the ring, again every 5 s
//!   while it lasts, and when it returns. Nothing is held for it: a call that waits for GPU work
//!   another submit brings would wait for ever. A call that took a quarter of a second and
//!   returned is logged too (video memory being paged shows there first);
//! - `XCL_STOP_FILE=<path>`, read by a thread of its own (`xcl-stop-file`) that calls nothing in
//!   the driver: when that file has content the title ends itself in order (`real::end_title`:
//!   its threads suspended, its fences signalled, the queues drained) – a process killed from
//!   outside with GPU work in flight can hang in the driver. A file that begins with `fault` (the
//!   driver reported an error) first gives the driver its time-out to say whether the device is
//!   removed, with the title's command lists held meanwhile; the watch thread's poll then writes
//!   the reason and DRED's report and ends the process.
//!
//! `XCL_GPU_WATCH=0`: no watch thread (the ring is still written, the stop file still read).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------
// the ring
// ---------------------------------------------------------------------------------------

/// draws and dispatches kept (about two frames of a retail title's world)
const RING: usize = 1 << 14;

static AT: AtomicUsize = AtomicUsize::new(0);
static RING_PSO: [AtomicUsize; RING] = [const { AtomicUsize::new(0) }; RING];
/// frame (28 bits) << 36 | work (32 bits) << 4 | kind
static RING_INFO: [AtomicU64; RING] = [const { AtomicU64::new(0) }; RING];

/// kinds of work in the ring (`gpu_profile::Kind` + 1, then the mesh draws; 4 bits)
pub(super) const MESH: u8 = 6;
pub(super) const INDIRECT_MESH: u8 = 7;
/// a DispatchMesh with a PC pipeline (a translated pass-through one)
pub(super) const PC_MESH: u8 = 8;

fn kind_name(kind: u8) -> &'static str {
    match kind {
        1 => "Dispatch",
        2 => "ExecuteIndirect (dispatches)",
        3 => "draw",
        4 => "ExecuteIndirect (draws)",
        5 => "ExecuteIndirect with a PC pipeline",
        MESH => "DispatchMesh in two passes",
        INDIRECT_MESH => "ExecuteIndirect (DispatchMesh) in two passes",
        PC_MESH => "DispatchMesh",
        _ => "?",
    }
}

fn dispatches(kind: u8) -> bool {
    kind == 1 || kind == 2
}

/// the product of a size's parts that are not 0 (a dispatch's groups, a draw's vertices x
/// instances, an ExecuteIndirect's largest count), at most `u32::MAX`
fn work_of(size: [u32; 3]) -> u32 {
    size.iter().filter(|&&d| d != 0).fold(1u64, |a, &d| a.saturating_mul(d as u64)).min(u32::MAX as u64) as u32
}

fn pack(kind: u8, work: u32, frame: u64) -> u64 {
    (frame & 0x0FFF_FFFF) << 36 | (work as u64) << 4 | (kind & 15) as u64
}

fn unpack(info: u64) -> (u8, u32, u32) {
    ((info & 15) as u8, (info >> 4) as u32, (info >> 36) as u32)
}

/// A draw or dispatch with pipeline `pso` (the title's handle) is recorded on a list.
pub(super) fn note(pso: usize, kind: u8, size: [u32; 3]) {
    let i = AT.fetch_add(1, Ordering::Relaxed) & (RING - 1);
    RING_PSO[i].store(pso, Ordering::Relaxed);
    RING_INFO[i].store(pack(kind, work_of(size), super::present::activity().0), Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    pso: usize,
    kind: u8,
    work: u32,
    frame: u32,
}

/// the ring, oldest first (entries never written left out)
fn snapshot() -> Vec<Entry> {
    let at = AT.load(Ordering::Relaxed);
    let n = at.min(RING);
    (at - n..at)
        .map(|k| {
            let i = k & (RING - 1);
            let (kind, work, frame) = unpack(RING_INFO[i].load(Ordering::Relaxed));
            Entry { pso: RING_PSO[i].load(Ordering::Relaxed), kind, work, frame }
        })
        .filter(|e| e.kind != 0)
        .collect()
}

/// The lines of a report of `entries` (oldest first): the dispatching pipelines by the groups
/// they were given, the drawing ones by their calls, and the last ones in order. `name`: what is
/// known of a pipeline.
fn summarise(entries: &[Entry], name: impl Fn(usize) -> String) -> Vec<String> {
    if entries.is_empty() {
        return vec!["recorded last: no draw or dispatch recorded yet".into()];
    }
    // (pipeline, kind) → calls, work in all, largest, last frame
    let mut by: std::collections::HashMap<(usize, u8), (u64, u64, u32, u32)> = std::collections::HashMap::new();
    for e in entries {
        let s = by.entry((e.pso, e.kind)).or_default();
        s.0 += 1;
        s.1 += e.work as u64;
        s.2 = s.2.max(e.work);
        s.3 = e.frame;
    }
    let (first, last) = (entries[0].frame, entries[entries.len() - 1].frame);
    let mut lines = vec![format!("recorded last: the last {} draws and dispatches (frames {first} to {last}), {} pipelines", entries.len(), by.len())];
    let mut all: Vec<((usize, u8), (u64, u64, u32, u32))> = by.into_iter().collect();
    // dispatches by the groups given (a program that runs long runs on every one of them),
    // then draws by calls; ties by address, for a stable order
    all.sort_by(|a, b| {
        dispatches(b.0 .1)
            .cmp(&dispatches(a.0 .1))
            .then(if dispatches(a.0 .1) { b.1 .1.cmp(&a.1 .1) } else { b.1 .0.cmp(&a.1 .0) })
            .then(a.0.cmp(&b.0))
    });
    let (mut shown_dispatch, mut shown_draw) = (0, 0);
    for ((pso, kind), (calls, work, largest, frame)) in &all {
        let shown = if dispatches(*kind) { &mut shown_dispatch } else { &mut shown_draw };
        *shown += 1;
        if *shown > if dispatches(*kind) { 24 } else { 12 } {
            continue;
        }
        lines.push(format!("recorded last:   {} {pso:#x} ({}): {calls} calls, {work} in all, largest {largest}, last in frame {frame}", kind_name(*kind), name(*pso)));
    }
    let tail: Vec<String> = entries[entries.len().saturating_sub(24)..].iter().map(|e| format!("{} {:#x} {}", kind_name(e.kind), e.pso, e.work)).collect();
    lines.push(format!("recorded last: the last {} in order: {}", tail.len(), tail.join("; ")));
    lines
}

/// What is known of the pipeline behind the title's handle `pso`: its console shaders (no side
/// effects: a report must not start a pipeline build). Graphics pipelines are in `pso_shaders`
/// only while profiling; DRED's label names them otherwise.
fn pipeline_name(pso: usize) -> String {
    let real = super::pso_async::peek(pso);
    match super::real::pso_shaders(real) {
        Some((cs, 0)) => format!("cs {cs:016x}"),
        Some((vs, ps)) => format!("vs {vs:016x} ps {ps:016x}"),
        None => super::dred::on().then(|| super::dred::label_of(real)).flatten().unwrap_or_else(|| "no console shader known".into()),
    }
}

/// The ring into the log (`why`: what happened), at most once a second.
pub(super) fn report(why: &str) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if last != 0 && now.saturating_sub(last) < 1000 {
        return;
    }
    LAST.store(now, Ordering::Relaxed);
    crate::xlog_error!("recorded last ({why}):");
    for line in summarise(&snapshot(), pipeline_name) {
        crate::xlog_error!("{line}");
    }
}

// ---------------------------------------------------------------------------------------
// calls into the driver
// ---------------------------------------------------------------------------------------

/// threads whose calls are timed (a title submits from a handful)
const SLOTS: usize = 64;

/// when the thread of slot k entered the driver (`now_ms`; 0: it is not in it)
static SINCE: [AtomicU64; SLOTS] = [const { AtomicU64::new(0) }; SLOTS];
static WHAT: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
static TID: [AtomicU32; SLOTS] = [const { AtomicU32::new(0) }; SLOTS];
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// this thread's slot (`usize::MAX`: none yet; `SLOTS`: none left)
    static SLOT: Cell<usize> = const { Cell::new(usize::MAX) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Call {
    Execute = 1,
    Present = 2,
}

fn call_name(what: u32) -> &'static str {
    match what {
        1 => "ExecuteCommandLists",
        2 => "Present",
        _ => "a call",
    }
}

/// milliseconds since the first use, never 0
fn now_ms() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// a call that returned after this long is logged
const SLOW_MS: u64 = 250;

/// The thread is inside a driver call until this is dropped.
pub(super) struct InDriver {
    slot: usize,
    since: u64,
}

/// The calling thread enters `call` (one the driver returns from at once when it works).
pub(super) fn in_driver(call: Call) -> InDriver {
    let slot = SLOT.with(|s| {
        if s.get() == usize::MAX {
            s.set(NEXT_SLOT.fetch_add(1, Ordering::Relaxed).min(SLOTS));
            if s.get() < SLOTS {
                TID[s.get()].store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }, Ordering::Relaxed);
            } else {
                static SAID: AtomicBool = AtomicBool::new(false);
                if !SAID.swap(true, Ordering::Relaxed) {
                    crate::xlog_warn!("D3D12: {SLOTS} threads have called into the driver – the calls of further threads are not timed");
                }
            }
        }
        s.get()
    });
    // (a call inside a call of the same thread is the outer one's time)
    if slot >= SLOTS || SINCE[slot].load(Ordering::Relaxed) != 0 {
        return InDriver { slot: SLOTS, since: 0 };
    }
    let since = now_ms();
    WHAT[slot].store(call as u32, Ordering::Relaxed);
    SINCE[slot].store(since, Ordering::Release);
    InDriver { slot, since }
}

impl Drop for InDriver {
    fn drop(&mut self) {
        if self.slot >= SLOTS {
            return;
        }
        SINCE[self.slot].store(0, Ordering::Release);
        let took = now_ms().saturating_sub(self.since);
        if took >= SLOW_MS {
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 16 || n.is_power_of_two() {
                crate::xlog_warn!("D3D12: {} took {took} ms in the driver ({n} such calls so far)", call_name(WHAT[self.slot].load(Ordering::Relaxed)));
            }
        }
    }
}

/// The call that has been in the driver longest: (milliseconds, which, thread).
fn longest_call(now: u64) -> Option<(u64, u32, u32)> {
    (0..SLOTS)
        .filter_map(|k| {
            let since = SINCE[k].load(Ordering::Acquire);
            (since != 0).then(|| (now.saturating_sub(since), WHAT[k].load(Ordering::Relaxed), TID[k].load(Ordering::Relaxed)))
        })
        .max()
}

/// a stop for a graphics driver error is being handled (`stop_asked`): the title's command lists
/// wait
static HOLD: AtomicBool = AtomicBool::new(false);

/// The title submits command lists: not while a stop for a driver error is being handled.
pub(super) fn wait_while_held() {
    wait_while(&HOLD);
}

/// `wait_while_held` on `hold`. A thread inside a timed call (a submit made from within it)
/// goes on: it would wait on itself.
fn wait_while(hold: &AtomicBool) {
    if !hold.load(Ordering::Acquire) || in_timed_call() {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed) < 8 {
        crate::xlog_warn!("D3D12: a submit is held – a stop for a graphics driver error is being handled");
    }
    while hold.load(Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The calling thread is inside a timed call (`in_driver`).
fn in_timed_call() -> bool {
    let slot = SLOT.with(|s| s.get());
    slot < SLOTS && SINCE[slot].load(Ordering::Relaxed) != 0
}

fn stall_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| std::env::var("XCL_GPU_STALL_MS").ok().and_then(|v| v.trim().parse().ok()).filter(|&ms| ms >= 100).unwrap_or(2000))
}

/// A call that stands still, as the watch knows it between polls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Stall {
    /// when it entered the driver (`now_ms`; 0: no call stands still)
    since: u64,
    /// when it was last logged
    said: u64,
}

/// What the watch logs after a poll.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Say {
    Nothing,
    /// a call passed the limit: (ms, which, thread)
    Stalled(u64, u32, u32),
    /// it still has not returned, 5 s after it was last logged: (ms, which, thread)
    Still(u64, u32, u32),
    /// it returned: how long it stood in the driver (ms)
    Returned(u64),
}

/// One poll of the watch: the stall before it, the time, the call in the driver longest
/// (`longest_call`) and the stall time in; the stall after it and what to log out.
fn watch_step(s: Stall, now: u64, longest: Option<(u64, u32, u32)>, limit: u64) -> (Stall, Say) {
    match longest {
        // (the call that stands still is the oldest in the driver until it returns)
        Some((ms, what, tid)) if s.since != 0 && now.saturating_sub(ms) == s.since => {
            if now.saturating_sub(s.said) >= 5000 {
                (Stall { said: now, ..s }, Say::Still(ms, what, tid))
            } else {
                (s, Say::Nothing)
            }
        }
        _ if s.since != 0 => (Stall::default(), Say::Returned(now.saturating_sub(s.since))),
        Some((ms, what, tid)) if ms >= limit => (Stall { since: now.saturating_sub(ms), said: now }, Say::Stalled(ms, what, tid)),
        _ => (s, Say::Nothing),
    }
}

/// the watch thread runs (`XCL_GPU_WATCH` is not 0)
static WATCHING: AtomicBool = AtomicBool::new(false);
/// the watch thread's heartbeat: when its last poll of the device returned (`now_ms`; when it
/// was started, before the first)
static BEAT: AtomicU64 = AtomicU64::new(0);

/// The watching thread and the stop file's, once the device is made.
pub(super) fn start() {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::AcqRel) {
        return;
    }
    start_stop_file();
    if std::env::var("XCL_GPU_WATCH").is_ok_and(|v| v.trim() == "0") {
        return;
    }
    BEAT.store(now_ms(), Ordering::Relaxed);
    let watch = std::thread::Builder::new().name("xcl-gpu-watch".into()).spawn(move || {
        let limit = stall_ms();
        let mut stall = Stall::default();
        loop {
            std::thread::sleep(Duration::from_millis(100));
            super::real::stop_if_device_removed();
            let now = now_ms();
            BEAT.store(now, Ordering::Relaxed);
            let (next, say) = watch_step(stall, now, longest_call(now), limit);
            stall = next;
            match say {
                Say::Nothing => {}
                Say::Stalled(ms, what, tid) => {
                    crate::xlog_warn!("D3D12: {} on thread {tid} has not returned for {ms} ms – a call the driver returns from at once when it works", call_name(what));
                    crate::log::flush();
                    report("a call into the driver stands still");
                    crate::log::flush();
                }
                Say::Still(ms, what, tid) => {
                    crate::xlog_error!("D3D12: {} on thread {tid} still has not returned ({ms} ms)", call_name(what));
                    crate::log::flush();
                }
                Say::Returned(ms) => {
                    crate::xlog_warn!("D3D12: the graphics driver answers again – the call returned after {ms} ms");
                    crate::log::flush();
                }
            }
        }
    });
    WATCHING.store(watch.is_ok(), Ordering::Release);
}

/// `XCL_STOP_FILE`: read on a thread of its own, which calls nothing in the driver or DXGI before
/// the title ends – a call that does not return (one on another thread never did in a driver
/// error) must not keep the stop from being taken.
fn start_stop_file() {
    let Some(path) = std::env::var_os("XCL_STOP_FILE").map(std::path::PathBuf::from) else { return };
    let _ = std::thread::Builder::new().name("xcl-stop-file".into()).spawn(move || {
        // polls in a row that found the file there and empty
        let mut empty = 0;
        loop {
            std::thread::sleep(Duration::from_millis(100));
            let Ok(bytes) = std::fs::read(&path) else {
                empty = 0;
                continue;
            };
            match stop_wanted(&bytes) {
                Some(fault) => stop_asked(fault),
                // (its writer may have made it and not written yet; one that stays empty: end)
                None => {
                    empty += 1;
                    if empty >= 10 {
                        stop_asked(false);
                    }
                }
            }
        }
    });
}

/// What a stop file's bytes ask for: None – nothing yet (empty or white space); Some(true) –
/// `fault…` (any case), the driver reported an error; Some(false) – anything else, end now.
/// UTF-8 (a BOM stripped) or UTF-16 LE (a BOM, or a NUL as the second byte), decoded lossily.
fn stop_wanted(bytes: &[u8]) -> Option<bool> {
    let text = if bytes.starts_with(&[0xFF, 0xFE]) || bytes.get(1) == Some(&0) {
        let units: Vec<u16> = bytes.strip_prefix(&[0xFF, 0xFE]).unwrap_or(bytes).chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        String::from_utf16_lossy(&units)
    } else {
        String::from_utf8_lossy(bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes)).into_owned()
    };
    let text = text.trim();
    (!text.is_empty()).then(|| text.get(..5).is_some_and(|w| w.eq_ignore_ascii_case("fault")))
}

/// A stop was asked for from outside (`fault`: the driver reported an error): the title ends
/// itself.
fn stop_asked(fault: bool) -> ! {
    crate::xlog_warn!("XCL_STOP_FILE: a stop was asked for from outside ({}) – the title ends itself", if fault { "the graphics driver reported an error" } else { "end" });
    crate::log::flush();
    report("a stop was asked for");
    crate::log::flush();
    if fault {
        // the driver gets its time-out to say whether the device is removed: the watch thread's
        // poll (`real::stop_if_device_removed`) then writes the reason and DRED's report and ends
        // the process; this thread calls nothing in the driver meanwhile. Only the title's
        // command lists are held (`q_execute_command_lists`): queue Signal and Wait, tile
        // mappings and pipeline creation go on. The hold stays into `end_title`, whose drain
        // signals and submits no list.
        HOLD.store(true, Ordering::Release);
        for _ in 0..120 {
            std::thread::sleep(Duration::from_millis(100));
        }
        // (since the last poll that returned, which may be before the stop was asked)
        let quiet = now_ms().saturating_sub(BEAT.load(Ordering::Relaxed));
        let check = if !WATCHING.load(Ordering::Acquire) {
            " – no check of the device runs (XCL_GPU_WATCH=0)".to_string()
        } else if quiet > 1000 {
            format!(" – the check of the device has itself not returned ({quiet} ms since the last one)")
        } else {
            String::new()
        };
        crate::xlog_warn!("XCL_STOP_FILE: the device was not reported removed within 12 s{check}");
        crate::log::flush();
    }
    crate::report::write_final(Some("stopped from outside (XCL_STOP_FILE)".into()));
    crate::log::flush();
    super::real::end_title(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_entry_packs_and_unpacks() {
        for (kind, work, frame) in [(1u8, 0u32, 0u64), (7, u32::MAX, 0x0FFF_FFFF), (3, 1234, 5), (PC_MESH, 9, 1)] {
            assert_eq!(unpack(pack(kind, work, frame)), (kind, work, frame as u32));
        }
        assert_eq!(kind_name(PC_MESH), "DispatchMesh");
        // a frame count past 28 bits wraps, and does not reach the work
        assert_eq!(unpack(pack(2, 7, 0x1_0000_0003)), (2, 7, 3));
        assert_eq!(work_of([4, 5, 6]), 120);
        assert_eq!(work_of([4, 0, 0]), 4, "an ExecuteIndirect's count");
        assert_eq!(work_of([0, 0, 0]), 1);
        assert_eq!(work_of([u32::MAX, u32::MAX, 2]), u32::MAX, "saturates");
    }

    #[test]
    fn the_report_ranks_dispatches_by_groups_and_draws_by_calls() {
        let e = |pso, kind, work, frame| Entry { pso, kind, work, frame };
        let entries = [
            e(0x10, 3, 6, 1),      // draw
            e(0x20, 1, 100, 1),    // small dispatch
            e(0x30, 1, 4_000_000, 2), // a full-screen dispatch
            e(0x10, 3, 6, 2),
            e(0x20, 1, 100, 2),
            e(0x40, MESH, 12, 2),
        ];
        let lines = summarise(&entries, |pso| format!("p{pso:x}"));
        assert_eq!(lines[0], "recorded last: the last 6 draws and dispatches (frames 1 to 2), 4 pipelines");
        assert!(lines[1].contains("Dispatch 0x30 (p30): 1 calls, 4000000 in all, largest 4000000, last in frame 2"), "{}", lines[1]);
        assert!(lines[2].contains("Dispatch 0x20 (p20): 2 calls, 200 in all, largest 100"), "{}", lines[2]);
        assert!(lines[3].contains("draw 0x10 (p10): 2 calls"), "{}", lines[3]);
        assert!(lines[4].contains("DispatchMesh in two passes 0x40 (p40): 1 calls"), "{}", lines[4]);
        assert!(lines[5].starts_with("recorded last: the last 6 in order: draw 0x10 6; Dispatch 0x20 100; "), "{}", lines[5]);
        assert_eq!(summarise(&[], |_| String::new()), vec!["recorded last: no draw or dispatch recorded yet".to_string()]);
    }

    #[test]
    fn the_report_keeps_to_a_few_pipelines_of_each_kind() {
        let entries: Vec<Entry> = (0..100).map(|k| Entry { pso: 0x1000 + k, kind: if k % 2 == 0 { 1 } else { 3 }, work: k as u32, frame: 1 }).collect();
        let lines = summarise(&entries, |_| String::new());
        // the head line, 24 dispatching pipelines, 12 drawing ones, the tail
        assert_eq!(lines.len(), 1 + 24 + 12 + 1);
        assert!(lines[1].contains("Dispatch 0x1062"), "the largest first: {}", lines[1]);
    }

    #[test]
    fn a_stall_is_logged_when_it_passes_the_limit_every_5_s_and_when_it_ends() {
        // a call enters the driver at 1 s and returns 13 s later; the watch polls every 100 ms
        let (limit, began) = (2000, 1000u64);
        let (mut stall, mut said) = (Stall::default(), Vec::new());
        for now in (1100..=20_000u64).step_by(100) {
            let longest = (now < began + 13_000).then(|| (now - began, Call::Execute as u32, 7));
            let (next, say) = watch_step(stall, now, longest, limit);
            stall = next;
            if say != Say::Nothing {
                said.push((now, say));
            }
        }
        assert_eq!(said, vec![(3000, Say::Stalled(2000, 1, 7)), (8000, Say::Still(7000, 1, 7)), (13_000, Say::Still(12_000, 1, 7)), (14_000, Say::Returned(13_000))]);
        assert_eq!(stall, Stall::default());
    }

    #[test]
    fn a_stall_ends_when_its_call_returns_and_another_is_in_the_driver() {
        let stalled = Stall { since: 1000, said: 3000 };
        assert_eq!(watch_step(Stall::default(), 3000, Some((1999, 1, 7)), 2000), (Stall::default(), Say::Nothing));
        assert_eq!(watch_step(Stall::default(), 3000, None, 2000), (Stall::default(), Say::Nothing));
        assert_eq!(watch_step(stalled, 4000, Some((3000, 1, 7)), 2000), (stalled, Say::Nothing), "the same call, said 1 s ago");
        assert_eq!(watch_step(stalled, 4000, Some((3, 2, 9)), 2000), (Stall::default(), Say::Returned(3000)), "that call returned, another began");
        // one past the limit already: logged at the next poll
        assert_eq!(watch_step(stalled, 6000, Some((2500, 2, 9)), 2000), (Stall::default(), Say::Returned(5000)));
        assert_eq!(watch_step(Stall::default(), 6100, Some((2600, 2, 9)), 2000), (Stall { since: 3500, said: 6100 }, Say::Stalled(2600, 2, 9)));
    }

    #[test]
    fn a_held_submit_goes_on_when_the_hold_ends_and_at_once_inside_a_timed_call() {
        // (holds of their own: other tests' submits are not held)
        static HELD: AtomicBool = AtomicBool::new(true);
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            tx.send(()).unwrap();
            wait_while(&HELD);
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(10)).expect("the thread runs");
        assert!(rx.recv_timeout(Duration::from_millis(300)).is_err(), "held");
        HELD.store(false, Ordering::Release);
        rx.recv_timeout(Duration::from_secs(10)).expect("goes on once the hold ends");
        t.join().unwrap();

        static HELD_TOO: AtomicBool = AtomicBool::new(true);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _call = in_driver(Call::Present);
            assert!(in_timed_call(), "a slot was left for this thread");
            wait_while(&HELD_TOO);
            let _ = tx.send(());
        });
        let inside = rx.recv_timeout(Duration::from_secs(10));
        HELD_TOO.store(false, Ordering::Release);
        inside.expect("a thread inside a timed call is not held (it would wait on itself)");
    }

    #[test]
    fn a_stop_file_says_whether_the_driver_reported_an_error() {
        assert_eq!(stop_wanted(b""), None, "made, not yet written");
        assert_eq!(stop_wanted(b" \r\n\t"), None);
        assert_eq!(stop_wanted(b"fault 2026-11-25T05:36:37\r\n"), Some(true));
        assert_eq!(stop_wanted(b"  FAULT"), Some(true));
        assert_eq!(stop_wanted(b"end"), Some(false));
        assert_eq!(stop_wanted(b"faul"), Some(false));
        assert_eq!(stop_wanted(b"\xEF\xBB\xBFfault\r\n"), Some(true), "a UTF-8 BOM");
        assert_eq!(stop_wanted(b"\xEF\xBB\xBF"), None);
        let utf16 = |s: &str, bom: bool| -> Vec<u8> { bom.then_some([0xFF, 0xFE]).into_iter().flatten().chain(s.encode_utf16().flat_map(u16::to_le_bytes)).collect() };
        for bom in [true, false] {
            assert_eq!(stop_wanted(&utf16("fault\r\n", bom)), Some(true), "UTF-16 LE, BOM {bom}");
            assert_eq!(stop_wanted(&utf16("  Fault", bom)), Some(true), "UTF-16 LE, BOM {bom}");
            assert_eq!(stop_wanted(&utf16("end\r\n", bom)), Some(false), "UTF-16 LE, BOM {bom}");
        }
        assert_eq!(stop_wanted(&utf16(" \r\n", true)), None);
    }

    #[test]
    fn a_thread_in_the_driver_is_seen_until_it_returns() {
        // (its own thread: the slot is the thread's)
        std::thread::spawn(|| {
            let tid = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
            let seen = |now: u64| (0..SLOTS).any(|k| TID[k].load(Ordering::Relaxed) == tid && SINCE[k].load(Ordering::Relaxed) != 0 && now >= SINCE[k].load(Ordering::Relaxed));
            {
                let _call = in_driver(Call::Execute);
                assert!(seen(now_ms()));
                // a call inside it is the outer one's time and does not end it
                drop(in_driver(Call::Present));
                assert!(seen(now_ms()));
                assert!(longest_call(now_ms() + 5000).is_some_and(|(ms, _, _)| ms >= 5000));
            }
            assert!(!seen(now_ms()));
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_ring_holds_what_was_noted_last() {
        // (other tests of the crate may note work too: only what this one put in is looked for)
        note(0xABCD_0001, MESH, [7, 0, 0]);
        note(0xABCD_0002, 1, [3, 5, 2]);
        let snap = snapshot();
        let a = snap.iter().rposition(|e| e.pso == 0xABCD_0001).expect("noted");
        let b = snap.iter().rposition(|e| e.pso == 0xABCD_0002).expect("noted");
        assert!(a < b, "oldest first");
        assert_eq!((snap[a].kind, snap[a].work), (MESH, 7));
        assert_eq!((snap[b].kind, snap[b].work), (1, 30));
    }
}
