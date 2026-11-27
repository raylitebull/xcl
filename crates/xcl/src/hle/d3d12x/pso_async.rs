//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A lazy pipeline (`lazy`) is built at its first use or ahead of it
//! (`Backlog`: every one, at the lowest priority). A draw or dispatch with a pipeline
//! still being built waits for it (`resolve_wait`) where it is used, a `Dispatch` at its submit
//! (`waits_for`, `need_by_submit`): a dispatch until it is built (a skipped one
//! corrupts what it computes) unless `XCL_CS_PSO_WAIT_MS` limits that, a draw only while its
//! thread has waited less than `XCL_GFX_PSO_WAIT_MS` in the last second when that is set
//! (`gfx_wait_left`) – then it and
//! the later draws of that pipeline draw nothing until it is built. An ExecuteIndirect whose
//! pipeline is still being built after that is skipped.
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    /// the driver pipeline binds get till then (`shared`); 0: the stand-in is one itself
    driver: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// a compute pipeline's console program (`fnv64` of its CS, as in the `[SHADER]` and driver
    /// compile lines); 0 = not known
    record: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    /// its dispatches may be left out while it is built (`image_only`): binds do not wait for it
    spared: std::sync::atomic::AtomicBool,
    demand: Mutex<Demand>,
    /// made by `submit_lazy` (the pipeline line counts these)
    lazy: bool,
    /// its build gave the driver a huge program (`note_huge`)
    huge: std::sync::atomic::AtomicBool,
    /// a wait for it passed 1 s and was logged (once a pipeline)
    warned: std::sync::atomic::AtomicBool,
    /// its translation and DXC compile on their own, until the backlog runs them (`Warm`)
    warm: Mutex<Option<Warm>>,
    /// what its first use tells besides `used::note` (`on_first_use`)
    on_use: Mutex<Option<OnUse>>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    /// binds waiting for it: a wait that ran out of its budget leaves (`Pending::unwant`), one
    /// that saw the build end stays counted
    waiters: u32,
    /// a call put off to the submit needs it (`need_by_submit`)
    by_submit: bool,
    /// a submit counts among `waiters` before it gets to it (`want`; left with the wait for it)
    ahead: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

impl Demand {
    fn wanted(&self) -> bool {
        self.waiters != 0
    }
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
    /// when the build this thread runs (`run_job`) started
    static JOB_START: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    /// the build this thread runs gave the driver a huge program (`note_huge`)
    static JOB_HUGE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// a backlog worker's build: whether it came from `Backlog::compute`; None once another
    /// worker took this one's place (`build_parks`)
    static PRELOAD_STEP: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `p`'s build `job` on this thread (a worker's, or a bind's that took it): its time and
/// whether its program was huge are noted for the logs.
fn run_job(p: &Pending, job: Job) -> usize {
    JOB_START.with(|s| s.set(Some(std::time::Instant::now())));
    JOB_HUGE.with(|h| h.set(false));
    let real = job();
    JOB_START.with(|s| s.set(None));
    if JOB_HUGE.with(|h| h.replace(false)) {
        p.huge.store(true, Ordering::Relaxed);
    }
    real
}

/// The time since the build this thread runs started (None: it runs none): the driver compile
/// log says where a slow pipeline's time went before the driver got it.
pub(super) fn job_elapsed() -> Option<std::time::Duration> {
    JOB_START.with(|s| s.get()).map(|t| t.elapsed())
}

/// The build this thread runs gives the driver a huge program (`real::huge_pso_bytes`).
pub(super) fn note_huge() {
    JOB_HUGE.with(|h| h.set(true));
}

/// This thread is a worker and one of the title's threads waits for what it is building: its
/// huge driver compile does not wait for free commit and may take one of the slots kept for
/// wanted builds, up to their cap (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted()))
}

/// This thread is a worker building a pipeline a bind waited for and went on without
/// (`resolve_within`): nobody waits for it now, but the title's draws or dispatches with it are
/// left out until it is built (`real::huge_compile_turn`).
pub(super) fn build_is_late() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.late.load(Ordering::Acquire)))
}

/// This thread is a backlog worker building a pipeline nobody has used yet: its huge driver
/// compile gives its turn to any other (`real::big_compile_turn`).
pub(super) fn build_is_ahead() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| !p.used.load(Ordering::Relaxed) && p.demand.lock().unwrap().preload_thread != 0))
}

/// The build ahead on this thread waits for its pipeline's first use before its driver compile
/// (`real::huge_compile_turn`: no giant one is made ahead), which may never come. On a backlog
/// worker another worker starts in its place and the lane of new compute pipelines goes to the
/// next one (`Backlog::built`), so the backlog does not stand still behind it; this thread ends
/// after the build.
pub(super) fn build_parks() {
    let Some(new_compute) = PRELOAD_STEP.with(|s| s.take()) else { return };
    let Some(q) = preload() else { return };
    q.list.lock().unwrap().built(new_compute);
    q.cv.notify_one();
    spawn_preload_worker(q);
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted());
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted() {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    complete(p, run_job(p, job));
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted() {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        Self::add_waiter(&mut self.demand.lock().unwrap());
    }

    fn add_waiter(d: &mut Demand) {
        d.waiters += 1;
        if d.waiters == 1 {
            Self::set_priority(d, windows_sys::Win32::System::Threading::THREAD_PRIORITY_NORMAL);
        }
    }

    /// A bind that waited for it ran out of its budget and goes on without it: when nobody else
    /// waits, the build is no longer wanted – back at the lowest priority on a preload worker and
    /// behind the huge-pipeline gate as an unwanted one (`real::big_compile_turn`): it waits for
    /// free commit and for the unwanted slots, where a wanted build skips the commit wait and has
    /// more slots, and many of those at once ran the process out of commit.
    fn unwant(&self) {
        let mut d = self.demand.lock().unwrap();
        d.waiters = d.waiters.saturating_sub(1);
        if d.waiters == 0 {
            Self::set_priority(&d, windows_sys::Win32::System::Threading::THREAD_PRIORITY_LOWEST);
        }
    }

    /// The wait that ran out of its budget was the one a submit had counted ahead (`want`): that
    /// one leaves too.
    fn unwant_ahead(&self) {
        let ahead = std::mem::take(&mut self.demand.lock().unwrap().ahead);
        if ahead {
            self.unwant();
        }
    }

    /// The preload worker building it, if one does, at `priority` (under the `demand` lock: the
    /// worker does not end its build meanwhile).
    fn set_priority(d: &Demand, priority: windows_sys::Win32::System::Threading::THREAD_PRIORITY) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_SET_LIMITED_INFORMATION};
        if d.preload_thread == 0 {
            return;
        }
        unsafe {
            let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
            if !h.is_null() {
                SetThreadPriority(h, priority);
                CloseHandle(h);
            }
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
            let on_use = self.on_use.lock().unwrap().take();
            if let Some(f) = on_use {
                f();
            }
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: super::sharded::Sharded<Arc<Pending>> = super::sharded::Sharded::new();

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    // (its translation ahead is of no use now: what it holds goes)
    drop(p.warm.lock().unwrap().take());
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
        if p.lazy {
            LAZY_BUILT.fetch_add(1, Ordering::Relaxed);
            if !p.used.load(Ordering::Relaxed) {
                LAZY_AHEAD.fetch_add(1, Ordering::Relaxed);
            }
            if p.huge.load(Ordering::Relaxed) {
                LAZY_HUGE.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// lazy pipelines (`submit_lazy`) made, of them graphics; built, of them before their first use,
/// and with a huge program
static LAZY_MADE: AtomicUsize = AtomicUsize::new(0);
static LAZY_GRAPHICS: AtomicUsize = AtomicUsize::new(0);
static LAZY_BUILT: AtomicUsize = AtomicUsize::new(0);
static LAZY_AHEAD: AtomicUsize = AtomicUsize::new(0);
static LAZY_HUGE: AtomicUsize = AtomicUsize::new(0);

/// What the pipeline line (`stats_line`) shows.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct Stats {
    made: usize,
    graphics: usize,
    built: usize,
    ahead: usize,
    huge: usize,
    /// translated ahead (`Backlog`), still in the backlog
    warmed: usize,
    backlog: usize,
    /// binds' waits for pipelines being built: (waits, µs) so far (`api_cost::Call::PipelineWait`)
    waits: (u64, u64),
}

fn stats() -> Stats {
    let l = |a: &AtomicUsize| a.load(Ordering::Relaxed);
    Stats {
        made: l(&LAZY_MADE),
        graphics: l(&LAZY_GRAPHICS),
        built: l(&LAZY_BUILT),
        ahead: l(&LAZY_AHEAD),
        huge: l(&LAZY_HUGE),
        warmed: l(&WARMED),
        backlog: l(&BACKLOG),
        waits: crate::hle::api_cost::totals()[crate::hle::api_cost::Call::PipelineWait as usize],
    }
}

/// The pipeline line for `now` when something changed since `last` (written 2 s before): how
/// many lazy pipelines the title made, how many are built and how many of those before their
/// first use (a bind did not wait for them), and how long binds stood still waiting for
/// pipelines in those 2 s (counted when a wait ends).
fn stats_line(now: &Stats, last: &Stats, huge_kb: usize) -> Option<String> {
    if now == last {
        return None;
    }
    Some(format!(
        "D3D12.X: lazy pipelines: {} made ({} graphics), {} built ({} before their first use, {} with {huge_kb} KB of DXIL or more), {} translated ahead, {} in the build-ahead backlog; binds waited {} ms for pipelines being built ({} waits in 2 s)",
        now.made,
        now.graphics,
        now.built,
        now.ahead,
        now.huge,
        now.warmed,
        now.backlog,
        (now.waits.1 - last.waits.1) / 1000,
        now.waits.0 - last.waits.0,
    ))
}

/// The pipeline line every 2 s while it changes (`stats_line`).
fn start_stats() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    // (a test's pipelines are not the title's)
    if cfg!(test) {
        return;
    }
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-pso-stats".into()).spawn(|| {
            let mut last = Stats::default();
            let mut last_put_off = PutOffStats::default();
            loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let now = stats();
                if let Some(line) = stats_line(&now, &last, super::real::huge_pso_bytes() >> 10) {
                    crate::xlog_info!("{line}");
                }
                last = now;
                let now = put_off_stats();
                if let Some(line) = put_off_line(&now, &last_put_off) {
                    crate::xlog_info!("{line}");
                }
                last_put_off = now;
            }
        });
    });
}

/// What became of a call put off to the submit (`list_x::defer`), for the line of them
/// (`put_off_line`).
#[derive(Clone, Copy)]
pub(super) enum PutOff {
    /// put off: its list split around it
    Deferred,
    /// its pipeline would have been waited for, but the list could not be split (no room for
    /// more segments, a predicated or bundle list, one that is no title list, `XCL_ALLOC_SWAP=0`):
    /// waited in the call
    Refused,
    /// recorded at a submit with its pipeline built
    Recorded,
    /// recorded at a submit, but its pipeline's build failed or ran out of its budget: skipped
    Skipped,
    /// not recorded at the submit (no allocator of the list's own, its Reset failed)
    Dropped,
}

static PUT_OFF: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
/// pipelines a put-off call needed (`need_by_submit`); of them built by a worker before its queue
static NEEDED_MADE: AtomicUsize = AtomicUsize::new(0);
static NEEDED_FIRST: AtomicUsize = AtomicUsize::new(0);
/// builds a thread that waits for the pipeline ran itself (`resolve_wait_for`); of them at a submit
static BUILT_HERE: AtomicUsize = AtomicUsize::new(0);
static BUILT_AT_SUBMIT: AtomicUsize = AtomicUsize::new(0);
/// waits at a submit for the pipelines of its put-off calls: count, µs
static SUBMIT_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SUBMIT_WAIT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn note_put_off(what: PutOff) {
    PUT_OFF[what as usize].fetch_add(1, Ordering::Relaxed);
    start_stats();
}

thread_local! {
    /// this thread records the put-off calls of a submit (`at_submit`)
    static AT_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While it lives this thread records a submit's put-off calls: its waits for their pipelines,
/// and the builds it runs itself, are the submit's (`put_off_line`).
pub(super) struct AtSubmit(bool);

pub(super) fn at_submit() -> AtSubmit {
    AtSubmit(AT_SUBMIT.with(|a| a.replace(true)))
}

impl Drop for AtSubmit {
    fn drop(&mut self) {
        AT_SUBMIT.with(|a| a.set(self.0));
    }
}

pub(super) fn is_at_submit() -> bool {
    AT_SUBMIT.with(|a| a.get())
}

/// How many put-off calls came to `what` so far.
#[cfg(test)]
pub(super) fn put_off_count(what: PutOff) -> usize {
    PUT_OFF[what as usize].load(Ordering::Relaxed)
}

/// What the line of put-off calls (`put_off_line`) shows.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
struct PutOffStats {
    /// by `PutOff`
    calls: [usize; 5],
    needed: usize,
    needed_first: usize,
    built_here: usize,
    built_at_submit: usize,
    /// every background build that ended (built or failed)
    builds: usize,
    /// waits at submits: (count, µs)
    waits: (u64, u64),
    /// the most segments a title list's recording had (`title_list::segments_peak`)
    segments: usize,
}

fn put_off_stats() -> PutOffStats {
    let l = |a: &AtomicUsize| a.load(Ordering::Relaxed);
    PutOffStats {
        calls: std::array::from_fn(|k| l(&PUT_OFF[k])),
        needed: l(&NEEDED_MADE),
        needed_first: l(&NEEDED_FIRST),
        built_here: l(&BUILT_HERE),
        built_at_submit: l(&BUILT_AT_SUBMIT),
        builds: l(&BUILT) + l(&FAILED),
        waits: (SUBMIT_WAITS.load(Ordering::Relaxed), SUBMIT_WAIT_US.load(Ordering::Relaxed)),
        segments: super::title_list::segments_peak(),
    }
}

/// The line of calls put off to the submit for `now` when it changed since `last` (written 2 s
/// before) and some call was put off or would have been: how many, what became of them at the
/// submits, the pipelines they needed and where those were built, and how long submits stood
/// in their waits for them.
fn put_off_line(now: &PutOffStats, last: &PutOffStats) -> Option<String> {
    let c = |k: PutOff| now.calls[k as usize];
    if now == last || c(PutOff::Deferred) + c(PutOff::Refused) == 0 {
        return None;
    }
    Some(format!(
        "D3D12.X: put-off dispatches: {} put off to the submit ({} in 2 s), {} waited in the call (no split); at submits {} recorded, {} skipped (build failed or late), {} dropped; {} pipelines needed by a submit ({} built by a worker before its queue); submits waited {} ms for them in 2 s ({} waits; {} ms so far); {} builds of {} on threads that waited for them ({} at a submit); at most {} segments in a recording",
        c(PutOff::Deferred),
        c(PutOff::Deferred) - last.calls[PutOff::Deferred as usize],
        c(PutOff::Refused),
        c(PutOff::Recorded),
        c(PutOff::Skipped),
        c(PutOff::Dropped),
        now.needed,
        now.needed_first,
        (now.waits.1 - last.waits.1) / 1000,
        now.waits.0 - last.waits.0,
        now.waits.1 / 1000,
        now.built_here,
        now.builds,
        now.built_at_submit,
        now.segments,
    ))
}
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// The PC calls of the title's threads (`pc_call`): the ones going on, and how long the ones
/// that ended took since the last frame was shown.
struct PcCalls {
    /// (id, when it began)
    now: Vec<(u64, std::time::Instant)>,
    /// the frames shown when `spent` began to count (`None` before the first observation)
    frame: Option<u64>,
    /// when the latest frame advance was first observed; calls that started earlier are clamped here
    frame_since: Option<std::time::Instant>,
    /// the time of the PC calls that ended since then
    spent: std::time::Duration,
}

impl PcCalls {
    const fn new() -> Self {
        Self { now: Vec::new(), frame: None, frame_since: None, spent: std::time::Duration::ZERO }
    }

    /// `frame` frames have been shown: the calls before the last of them are not this wait's.
    fn at(&mut self, frame: u64, now: std::time::Instant) {
        match self.frame {
            None => self.frame = Some(frame),
            Some(prev) if frame > prev => {
                self.frame = Some(frame);
                self.frame_since = Some(now);
                self.spent = std::time::Duration::ZERO;
            }
            Some(_) => {}
        }
    }

    fn clamp(&self, since: std::time::Instant) -> std::time::Instant {
        self.frame_since.map_or(since, |f| since.max(f))
    }

    fn begin(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        self.now.push((id, now));
    }

    fn end(&mut self, id: u64, frame: u64, now: std::time::Instant) {
        self.at(frame, now);
        if let Some(i) = self.now.iter().position(|c| c.0 == id) {
            let (_, since) = self.now.swap_remove(i);
            self.spent += now.saturating_duration_since(self.clamp(since));
        }
    }

    /// Whether a call going on holds the title: the PC calls since the last frame, it included,
    /// have taken `PC_CALL_HOLDS_AFTER`.
    fn holds(&mut self, frame: u64, now: std::time::Instant) -> bool {
        self.at(frame, now);
        let spent = self.spent;
        self.now.iter().any(|(_, since)| spent + now.saturating_duration_since(self.clamp(*since)) >= PC_CALL_HOLDS_AFTER)
    }
}

static PC_CALLS: Mutex<PcCalls> = Mutex::new(PcCalls::new());
/// From this much time in PC calls since the last frame on, a PC call holds the title: the
/// console's calls return at once.
const PC_CALL_HOLDS_AFTER: std::time::Duration = std::time::Duration::from_millis(250);

/// Whether one of the title's threads is held where a console title never waits: a bind is
/// waiting for a pipeline still being built, or a PC call is going on after such calls have
/// kept the title's threads for long (`pc_call`). `kernelx::Sleep` does not count that time.
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0 || held_by_pc_call()
}

fn held_by_pc_call() -> bool {
    PC_CALLS.lock().unwrap().holds(super::present::presents(), std::time::Instant::now())
}

/// A call of the title into D3D12 that XCL or the PC's driver can keep for long where the
/// console's takes no time: a submission (XCL's uploads and tile mappings before the lists, the
/// driver's compiling in them), a texture's placement, a pipeline built in the call. While it
/// lasts the title is held (`title_held`), once the PC calls since the last frame shown, this
/// one included, have taken `PC_CALL_HOLDS_AFTER`: one long call, or many short ones in a row
/// (a retail title placed 16 384 textures in 29 s without a frame, 1 - 50 ms each). A title
/// that shows its frames is not held by its submissions.
pub struct PcCall(u64);

pub fn pc_call() -> PcCall {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PC_CALLS.lock().unwrap().begin(id, super::present::presents(), std::time::Instant::now());
    PcCall(id)
}

impl Drop for PcCall {
    fn drop(&mut self) {
        PC_CALLS.lock().unwrap().end(self.0, super::present::presents(), std::time::Instant::now());
    }
}

/// Whether this thread is building a pipeline for `build` (a worker, not one of the title's).
pub(super) fn building_here() -> bool {
    BUILDING.with(|b| b.borrow().is_some())
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

/// pipelines a submit will wait for: the workers take these before their queue (`need_by_submit`)
static NEEDED: Mutex<std::collections::VecDeque<Arc<Pending>>> = Mutex::new(std::collections::VecDeque::new());

fn next_needed() -> Option<Arc<Pending>> {
    NEEDED.lock().unwrap().pop_front()
}

/// Drop an unstarted orphaned job outside all locks and wake any concurrent waiter.
fn drop_unstarted_job(p: &Pending) {
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        {
            let mut done = p.done.lock().unwrap();
            *done = true;
            p.cv.notify_all();
        }
        drop(job);
        // (its translation ahead is of no use either, as in `complete`: the backlog may hold
        // the entry a while yet)
        drop(p.warm.lock().unwrap().take());
    }
}

type Work = Box<dyn FnOnce() + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Work>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Work>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Work>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(work) = next else { return };
                while let Some(p) = next_needed() {
                    if build_queued(&p) {
                        NEEDED_FIRST.fetch_add(1, Ordering::Relaxed);
                    }
                }
                work();
            });
        }
        Mutex::new(tx)
    })
}

fn enqueue(p: Arc<Pending>) {
    let _ = queue().lock().unwrap().send(Box::new(move || {
        build_queued(&p);
    }));
}

/// Whether this worker built `p` (not: released, or another thread took the build).
fn build_queued(p: &Arc<Pending>) -> bool {
    // (released by the title meanwhile)
    if p.orphan.load(Ordering::Acquire) {
        drop_unstarted_job(p);
        return false;
    }
    // (None: a bind took it and builds it itself)
    let job = p.job.lock().unwrap().take();
    let Some(job) = job else { return false };
    build(p, job, false);
    true
}

/// This thread is a backlog worker whose build nobody waits for (yet).
fn build_is_backlog_only() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| {
        let d = p.demand.lock().unwrap();
        d.preload_thread != 0 && !d.wanted()
    }))
}

/// The caller also drains its batch. Multiple workers can therefore build
/// different chains without ever waiting for a free worker to start a piece.
/// A backlog worker's build nobody waits for keeps its pieces on its own thread: helpers would go
/// to the workers whose builds binds wait for (`queue`), ahead of them.
pub(super) fn build_pieces(jobs: Vec<Job>) -> Vec<usize> {
    let count = jobs.len();
    let jobs = Arc::new(Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>()));
    let (send, recv) = mpsc::channel();
    let drain = move || loop {
        let job = jobs.lock().unwrap().pop();
        let Some((i, job)) = job else { return };
        // (a piece that panics fails: its result is still sent, or the caller would wait for ever)
        let _ = send.send((i, std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).unwrap_or(0)));
    };
    // the helpers build for this worker's pipeline: the title waiting for it waits for them too
    let parent = BUILDING.with(|b| b.borrow().clone());
    let width = if build_is_backlog_only() { 1 } else { count.min(crate::gpu::recomp::dxc::compile_threads()) };
    for _ in 1..width {
        let (drain, parent) = (drain.clone(), parent.clone());
        let _ = queue().lock().unwrap().send(Box::new(move || {
            let before = BUILDING.with(|b| b.replace(parent));
            drain();
            BUILDING.with(|b| *b.borrow_mut() = before);
        }));
    }
    drain();
    let mut results = vec![0; count];
    for (i, result) in recv.iter().take(count) { results[i] = result; }
    results
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_with(standin, compute, job, 0);
}

/// `submit`, with a compute pipeline's console program for the log (`record`; 0: not known).
pub(super) fn submit_with(standin: usize, compute: bool, job: Job, record: u64) {
    submit_as(standin, compute, job, true, 0, None, record);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind – or ahead of that by the
/// lowest-priority workers (`preload`, `Backlog`). A retail game makes ~51 000 pipelines in
/// its first 30 s and uses few; each one built stays in the driver (~400 KB), and the process
/// ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – and built ahead of it
/// at the lowest priority (`Backlog`): first when an earlier run used a pipeline of these
/// shaders (`key`; `used`), after those one of any other shaders (`Backlog` has the order).
/// `key` 0: only at its first use.
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_lazy_with(standin, compute, job, key, None, 0);
}

/// `submit_lazy`, with the pipeline's translation and DXC compile on their own (`warm`): the
/// backlog does that for every pipeline it holds before the driver builds any of them. `record`:
/// a compute pipeline's console program, for the log (0: not known).
pub(super) fn submit_lazy_with(standin: usize, compute: bool, job: Job, key: u64, warm: Option<Warm>, record: u64) {
    submit_as(standin, compute, job, false, key, warm, record);
}

/// A lazy pipeline's translation and DXC compile without the driver's: what makes the DXIL
/// caches (in memory and on disk) hold its shaders, so its build is the driver's compile only.
pub type Warm = Box<dyn FnOnce() + Send>;

impl Pending {
    fn new(job: Job, send: bool, compute: bool, key: u64, warm: Option<Warm>, record: u64) -> Pending {
        Pending {
            job: Mutex::new(Some(job)),
            sent: std::sync::atomic::AtomicBool::new(send),
            real: AtomicUsize::new(0),
            driver: AtomicUsize::new(0),
            done: Mutex::new(false),
            cv: Condvar::new(),
            compute,
            key,
            record,
            used: Default::default(),
            orphan: Default::default(),
            late: Default::default(),
            spared: Default::default(),
            demand: Default::default(),
            lazy: !send,
            huge: Default::default(),
            warned: Default::default(),
            warm: Mutex::new(warm),
            on_use: Default::default(),
        }
    }

    /// Nobody has taken it yet: not sent to the workers (a bind wants it: they build it), not
    /// released, not being built.
    fn unclaimed(&self) -> bool {
        !self.sent.load(Ordering::Acquire) && self.untaken()
    }

    /// Not released and not being built (sent or not).
    fn untaken(&self) -> bool {
        !self.orphan.load(Ordering::Acquire) && self.job.lock().unwrap().is_some()
    }
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64, warm: Option<Warm>, record: u64) {
    let p = Arc::new(Pending::new(job, send, compute, key, warm, record));
    if !send {
        LAZY_MADE.fetch_add(1, Ordering::Relaxed);
        if !compute {
            LAZY_GRAPHICS.fetch_add(1, Ordering::Relaxed);
        }
        start_stats();
    }
    PENDING.insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
        return;
    }
    let known = used::known(key);
    if !known && (key == 0 || ahead_cap() == 0) {
        return;
    }
    if let Some(q) = preload() {
        let mut b = q.list.lock().unwrap();
        b.push(p, known);
        BACKLOG.store(b.len(), Ordering::Relaxed);
        q.cv.notify_one();
    }
}

/// Run on the thread of a lazy pipeline's first use, once (it is to hand work to another).
pub type OnUse = Box<dyn FnOnce() + Send>;

/// What the first use of lazy pipeline `standin` tells besides `used::note`: a graphics one's
/// description for the next run's preparation (`gfx_records::note`). Told before the title has
/// the stand-in.
pub(super) fn on_first_use(standin: usize, f: OnUse) {
    if let Some(p) = entry(standin) {
        *p.on_use.lock().unwrap() = Some(f);
    }
}

/// The shaders of a pipeline's set, told when the set's key is made (`real::OwnedDesc`).
pub(crate) fn shaders_of(key: u64, shaders: Vec<u64>) {
    used::parts(key, shaders);
}

/// The shaders of the pipelines earlier runs used, each by its own key (`used-shaders.txt`
/// beside the DXIL cache): what a title's shader preparation translates. Empty before a run
/// with this list has used a pipeline.
pub fn earlier_used_shaders() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-shaders.txt")).unwrap_or_default())
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use (also ahead of it, after those of earlier runs:
/// `Backlog`): until then it is waited for on the title's thread. A retail
/// title's first frames of its world use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
        /// the shaders of each set made in this run, by the set's key; the shaders of the sets
        /// used so far and `used-shaders.txt`, where they are kept for the next run
        parts: Mutex<std::collections::HashMap<u64, Vec<u64>>>,
        shaders: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)), parts: Default::default(), shaders: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            let shaders_path = path.with_file_name("used-shaders.txt");
            let shaders = parse(&std::fs::read_to_string(&shaders_path).unwrap_or_default());
            let shaders_file = std::fs::OpenOptions::new().create(true).append(true).open(&shaders_path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier), parts: Default::default(), shaders: Mutex::new((shaders, shaders_file)) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        {
            let mut g = get().now.lock().unwrap();
            if g.0.insert(key) {
                if let Some(f) = g.1.as_mut() {
                    let _ = writeln!(f, "{key:016x}");
                }
            }
        }
        // its shaders, each once (also those of a set an earlier run used: the list of
        // shaders is younger than the list of sets)
        let parts = get().parts.lock().unwrap().get(&key).cloned().unwrap_or_default();
        let mut g = get().shaders.lock().unwrap();
        for part in parts {
            if g.0.insert(part) {
                if let Some(f) = g.1.as_mut() {
                    let _ = writeln!(f, "{part:016x}");
                }
            }
        }
    }

    /// The shaders of the set `key`, each by its own key (`real::shader_key`).
    pub fn parts(key: u64, shaders: Vec<u64>) {
        if key != 0 {
            get().parts.lock().unwrap().entry(key).or_insert(shaders);
        }
    }

    /// (tests) This run used the shader `key`.
    #[cfg(test)]
    pub fn shader_noted(key: u64) -> bool {
        get().shaders.lock().unwrap().0.contains(&key)
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build lazy pipelines ahead of their first use (`XCL_PSO_PRELOAD=0`: off),
/// in the order of `Backlog`. Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it; a bind's wait raises the worker building it), and the title's own
/// threads are not either.
struct Preload {
    list: Mutex<Backlog>,
    cv: Condvar,
}

/// The lazy pipelines to build ahead of their first use, in this order: the compute ones of
/// shaders an earlier run used (`used`); the title's other compute pipelines; the graphics ones
/// of shaders an earlier run used; then the title's other graphics pipelines, first translated
/// and compiled by DXC all of them (`Warm`: milliseconds to seconds each, no driver memory kept,
/// and the disk caches hold them for the next run), then built by the driver one by one. A
/// retail title made its ~2 700 pipelines in its first 6 s and bound some of them for the first
/// time up to 150 s later: each such bind waited 7 – 15 s for its pipeline, one after another,
/// and no frame came for 156 s. A compute pipeline is waited for at its first dispatch (or its
/// dispatches are skipped: `XCL_CS_PSO_WAIT_MS`), and one no earlier run used was built only
/// then: a retail title's thread spent 57 s in the driver on one. Those are built by one worker
/// at a time (`compute_out`): a title has ~1 600 of them, some a minute each in the driver, and
/// with every worker on them the known graphics pipelines would wait for all of them and be
/// built at their first draws instead. A pipeline a bind wants is
/// not taken from here (`unclaimed`): the workers build it at once, and a huge driver compile
/// ahead of its use gives its turn to any other and takes no giant slot (`real::big_compile_turn`).
#[derive(Default)]
struct Backlog {
    known_compute: std::collections::VecDeque<Arc<Pending>>,
    /// compute pipelines no earlier run used
    compute: std::collections::VecDeque<Arc<Pending>>,
    /// a worker builds one of `compute` or `spared` (`next` hands out no other until `built`)
    compute_out: bool,
    /// compute pipelines binds go on without while they are built (`image_only`), sent here at
    /// their first bind (`send_spared`): used, so built before `compute` and with no room or cap
    spared: std::collections::VecDeque<Arc<Pending>>,
    known: std::collections::VecDeque<Arc<Pending>>,
    warm: std::collections::VecDeque<Arc<Pending>>,
    cold: std::collections::VecDeque<Arc<Pending>>,
}

/// What a backlog worker does with a pipeline.
enum Step {
    /// its build (`ahead`: of a pipeline no earlier run used – counts against `ahead_cap`;
    /// `new_compute`: from `Backlog::compute`, `Backlog::built` after it)
    Build { ahead: bool, new_compute: bool },
    /// its translation and DXC compile; then back to the backlog for its build
    Warm(Warm),
}

impl Backlog {
    fn push(&mut self, p: Arc<Pending>, known: bool) {
        if known && p.compute {
            self.known_compute.push_back(p);
        } else if p.compute {
            self.compute.push_back(p);
        } else if known {
            self.known.push_back(p);
        } else if p.warm.lock().unwrap().is_some() {
            self.warm.push_back(p);
        } else {
            self.cold.push_back(p);
        }
    }

    fn len(&self) -> usize {
        self.known_compute.len() + self.compute.len() + self.spared.len() + self.known.len() + self.warm.len() + self.cold.len()
    }

    /// The next piece of work. `ahead`: pipelines no earlier run used may be worked on now
    /// (`XCL_PSO_AHEAD`, free commit: `ahead_room`); `builds`: one more of them may be built by
    /// the driver (`ahead_cap`). Without them those wait here.
    fn next(&mut self, ahead: bool, builds: bool) -> Option<(Arc<Pending>, Step)> {
        while let Some(p) = self.known_compute.pop_front() {
            if p.unclaimed() {
                return Some((p, Step::Build { ahead: false, new_compute: false }));
            }
        }
        // (`sent`: its copy in another lane is passed over there)
        if !self.compute_out {
            while let Some(p) = self.spared.pop_front() {
                if p.untaken() {
                    self.compute_out = true;
                    return Some((p, Step::Build { ahead: false, new_compute: true }));
                }
            }
        }
        if ahead && builds && !self.compute_out {
            while let Some(p) = self.compute.pop_front() {
                if p.unclaimed() {
                    self.compute_out = true;
                    return Some((p, Step::Build { ahead: true, new_compute: true }));
                }
            }
        }
        while let Some(p) = self.known.pop_front() {
            if p.unclaimed() {
                return Some((p, Step::Build { ahead: false, new_compute: false }));
            }
        }
        if !ahead {
            return None;
        }
        while let Some(p) = self.warm.pop_front() {
            if !p.unclaimed() {
                continue;
            }
            let warm = p.warm.lock().unwrap().take();
            match warm {
                Some(w) => return Some((p, Step::Warm(w))),
                None => self.cold.push_back(p),
            }
        }
        if !builds {
            return None;
        }
        while let Some(p) = self.cold.pop_front() {
            if p.unclaimed() {
                return Some((p, Step::Build { ahead: true, new_compute: false }));
            }
        }
        None
    }

    /// The build `next` handed out as `Step::Build` is over (built, or taken by a bind).
    fn built(&mut self, new_compute: bool) {
        if new_compute {
            self.compute_out = false;
        }
    }
}

/// Lazy pipelines no earlier run used that the driver builds ahead of their first use
/// (`XCL_PSO_AHEAD`, default 4096; 0: none, and none translated ahead either – only those of
/// earlier runs are built ahead then). Each pipeline built keeps ~400 KB in the driver whether
/// the title uses it or not: a retail game made ~51 000 and used few.
fn ahead_cap() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| ahead_cap_of(std::env::var("XCL_PSO_AHEAD").ok().as_deref()))
}

fn ahead_cap_of(v: Option<&str>) -> usize {
    v.and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(4096)
}

/// Commit that must be free for building ahead pipelines no earlier run used: the 6 GB DXC's
/// compiles keep free for a running title (`dxc::wait_for_memory`) and 2 GB more. Below it they
/// wait in the backlog (pipelines of an earlier run are built as before).
const AHEAD_FREE: u64 = 8 << 30;

fn ahead_room() -> bool {
    ahead_cap() > 0 && super::real::big_compile_commit_free(AHEAD_FREE)
}

/// pipelines no earlier run used: translated ahead, built ahead by the backlog (`ahead_cap`);
/// pipelines in the backlog
static WARMED: AtomicUsize = AtomicUsize::new(0);
static AHEAD_BUILDS: AtomicUsize = AtomicUsize::new(0);
static BACKLOG: AtomicUsize = AtomicUsize::new(0);

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for _ in 0..n {
            spawn_preload_worker(q);
        }
    });
    Some(q)
}

/// One more worker of the backlog (`preload`).
fn spawn_preload_worker(q: &'static Preload) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let k = NEXT.fetch_add(1, Ordering::Relaxed);
    let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
        }
        static DONE: AtomicUsize = AtomicUsize::new(0);
        loop {
            let (p, step) = {
                let mut g = q.list.lock().unwrap();
                loop {
                    let next = g.next(ahead_room(), AHEAD_BUILDS.load(Ordering::Relaxed) < ahead_cap());
                    BACKLOG.store(g.len(), Ordering::Relaxed);
                    if let Some(next) = next {
                        break next;
                    }
                    // (pipelines waiting for free commit are looked at again in a second)
                    g = q.cv.wait_timeout(g, std::time::Duration::from_secs(1)).unwrap().0;
                }
            };
            let (ahead, new_compute) = match step {
                Step::Warm(warm) => {
                    warm();
                    WARMED.fetch_add(1, Ordering::Relaxed);
                    if p.unclaimed() {
                        q.list.lock().unwrap().cold.push_back(p);
                    }
                    continue;
                }
                Step::Build { ahead, new_compute } => (ahead, new_compute),
            };
            // (None: a bind took it and builds it itself)
            let job = p.job.lock().unwrap().take();
            if let Some(job) = job {
                if ahead {
                    AHEAD_BUILDS.fetch_add(1, Ordering::Relaxed);
                }
                PRELOAD_STEP.with(|s| s.set(Some(new_compute)));
                build(&p, job, true);
                let parked = PRELOAD_STEP.with(|s| s.take()).is_none();
                let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                }
                // (its lane went back and another worker took its place: `build_parks`)
                if parked {
                    return;
                }
            }
            q.list.lock().unwrap().built(new_compute);
        }
    });
}

#[cfg(test)]
mod used_tests {
    use super::*;

    /// The shaders of a set are kept when a pipeline of the set is used, each once, and not
    /// before: a set that is only made keeps nothing.
    #[test]
    fn the_shaders_of_a_used_set_are_kept_for_the_next_run() {
        let (set, other) = (0x5e70_0000_0000_0001u64, 0x5e70_0000_0000_0003u64);
        let (vs, ps, cs) = (0x5e70_0000_0000_0011u64, 0x5e70_0000_0000_0013u64, 0x5e70_0000_0000_0015u64);
        shaders_of(set, vec![vs, ps]);
        shaders_of(other, vec![cs]);
        assert!(!used::shader_noted(vs) && !used::shader_noted(ps), "made, not used yet");
        used::note(set);
        assert!(used::shader_noted(vs) && used::shader_noted(ps), "the shaders of the used set");
        assert!(!used::shader_noted(cs), "the other set was not used");
    }

    /// A PC call that has kept one of the title's threads for long holds the title, a short one
    /// does not, and the hold ends with the call.
    #[test]
    fn a_long_pc_call_holds_the_title() {
        let mut calls = PcCalls::new();
        let began = std::time::Instant::now();
        calls.begin(1, 0, began);
        calls.begin(2, 0, began + 2 * PC_CALL_HOLDS_AFTER);
        assert!(!calls.holds(0, began), "a call that has just begun holds the title");
        let later = began + 2 * PC_CALL_HOLDS_AFTER;
        assert!(calls.holds(0, later), "a PC call has lasted {:?}, but the title is not held", 2 * PC_CALL_HOLDS_AFTER);
        calls.end(1, 0, later);
        calls.end(2, 0, later);
        assert!(!calls.holds(0, later), "the calls are over, the title is still held");
    }

    /// Short PC calls add up between two frames: none of them is long, but after many in a row
    /// the one going on holds the title. The title's own time between them is not held, and a
    /// frame shown ends it.
    #[test]
    fn short_pc_calls_between_two_frames_hold_the_title() {
        let mut calls = PcCalls::new();
        let step = PC_CALL_HOLDS_AFTER / 10;
        let mut now = std::time::Instant::now();
        for id in 0..20 {
            calls.begin(id, 7, now);
            assert!(id >= 9 || !calls.holds(7, now + step), "{id} short calls hold the title already");
            now += step;
            calls.end(id, 7, now);
        }
        assert!(!calls.holds(7, now), "no PC call is going on, the title is held");
        calls.begin(20, 7, now);
        assert!(calls.holds(7, now), "twenty short PC calls since the last frame and one going on, but the title is not held");
        assert!(!calls.holds(8, now), "a frame was shown, the calls before it still hold the title");
        calls.end(20, 8, now);
    }

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A stand-in that is no driver object (`shared`) never reaches the driver: binds get the
    /// driver pipeline shared with other stand-ins until the real one is built, and after a build
    /// that failed.
    #[test]
    fn a_bind_of_a_shared_stand_in_gets_its_driver_pipeline() {
        let (fails, builds, driver) = (0x7E57_B011usize, 0x7E57_B012usize, 0x7E57_B013usize);
        submit_lazy(fails, false, Box::new(|| 0), 0);
        submit_lazy(builds, false, Box::new(|| 0x1234), 0);
        for s in [fails, builds] {
            shared(s, driver);
            assert_eq!(entry(s).unwrap().stand(s), driver);
        }
        assert_eq!(resolve_wait_for(fails, None), driver, "its build failed");
        assert_eq!(resolve(fails), driver);
        assert_eq!(resolve_wait_for(builds, None), 0x1234);
        assert_eq!(resolve_for_bind(builds), 0x1234);
        // (0x1234 is no pipeline: taken, not released)
        assert_eq!((released(fails), released(builds)), (0, 0x1234));
    }

    /// A bind does not wait for a compute pipeline that writes only images: it gets the stand-in
    /// at once, with no budget, the build goes to the workers
    /// unwanted, and once built the bind gets the real one. Without `image_only` the same bind
    /// waits until the pipeline is built.
    #[test]
    fn a_bind_does_not_wait_for_a_pipeline_that_writes_only_images() {
        let (spared, waited) = (0x7E57_B0A1usize, 0x7E57_B0A2usize);
        let (go, finish) = mpsc::channel::<()>();
        let (tx, rx) = mpsc::channel();
        submit_lazy(spared, true, Box::new(move || {
            let _ = tx.send(wanted_here()());
            let _ = finish.recv();
            0x5A5A
        }), 0);
        image_only(spared);
        // (on a thread: a bind that waits would wait for `go`, below)
        let (bound_tx, bound) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = bound_tx.send((resolve_wait_for(spared, None), resolve_wait_for(spared, None)));
        });
        let binds = bound.recv_timeout(std::time::Duration::from_secs(5));
        if binds.is_err() {
            let _ = go.send(());
        }
        assert_eq!(binds.expect("the binds waited for the build"), (spared, spared), "still being built: the stand-in");
        assert!(is_pending(spared));
        assert!(!rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it"), "nobody waits: the build is not wanted");
        go.send(()).unwrap();
        let start = std::time::Instant::now();
        while is_pending(spared) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(spared, None), 0x5A5A, "built by a worker");
        // the same build without the mark: the bind waits for it
        submit_lazy(waited, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            0x5B5B
        }), 0);
        assert_eq!(resolve_wait_for(waited, None), 0x5B5B);
        let _ = released(spared);
        let _ = released(waited);
    }

    #[test]
    fn a_pipeline_a_submit_needs_is_built_before_the_queue() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let backlog = 8 * crate::gpu::recomp::dxc::compile_threads();
        let standins: Vec<usize> = (0..=backlog).map(|i| 0x7E57_E000usize + i).collect();
        for &standin in &standins {
            let order = order.clone();
            submit(standin, true, Box::new(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                order.lock().unwrap().push(standin);
                0x6161
            }));
        }
        let last = *standins.last().unwrap();
        need_by_submit(last);
        let start = std::time::Instant::now();
        while standins.iter().any(|&s| is_pending(s)) && start.elapsed() < std::time::Duration::from_secs(30) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let order = order.lock().unwrap();
        assert_eq!(order.len(), standins.len());
        let at = order.iter().position(|&s| s == last).unwrap();
        assert!(at < order.len() / 2, "made last and needed by a submit, it was built {at} of {}", order.len());
        for &standin in &standins {
            let _ = released(standin);
        }
    }

    /// A dispatch is put off to the submit only where it would wait now: not with a pipeline that
    /// writes only images (its dispatches are left out while it is built) nor with one a bind
    /// already waited for past its budget. A submit that says it will wait for a pipeline counts
    /// once among its waiters however often it says so, and leaves when that wait runs out; a
    /// late or spared pipeline it does not count at all.
    #[test]
    fn a_dispatch_is_put_off_only_where_it_would_wait() {
        let (plain, spared) = (0x7E57_F001usize, 0x7E57_F002usize);
        let (go, finish) = mpsc::channel::<()>();
        let finish = Arc::new(Mutex::new(finish));
        for standin in [plain, spared] {
            let finish = finish.clone();
            submit_lazy(standin, true, Box::new(move || {
                let _ = finish.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10));
                0x7171
            }), 0);
        }
        image_only(spared);
        assert!(waits_for(plain));
        assert!(!waits_for(spared), "its dispatches are left out while it is built");
        let p = entry(plain).unwrap();
        want(plain);
        want(plain);
        assert_eq!(p.demand.lock().unwrap().waiters, 1, "a submit counts once");
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.demand.lock().unwrap().waiters, 0, "the submit's count leaves with the wait that ran out");
        assert!(!waits_for(plain), "late: binds go on without it");
        // a later submit that puts it off does not wait for it: it does not count
        want(plain);
        assert_eq!(p.demand.lock().unwrap().waiters, 0, "a late build stays unwanted");
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.demand.lock().unwrap().waiters, 0);
        // a submit counted before another thread's wait made it late leaves at its own
        {
            let mut d = p.demand.lock().unwrap();
            d.ahead = true;
            Pending::add_waiter(&mut d);
        }
        assert_eq!(resolve_within(&p, plain, std::time::Duration::from_millis(50)), plain);
        assert_eq!(p.demand.lock().unwrap().waiters, 0, "the submit's count leaves with its wait for a late build");
        want(spared);
        assert_eq!(entry(spared).unwrap().demand.lock().unwrap().waiters, 0, "nobody waits for a spared build");
        drop(go);
        let start = std::time::Instant::now();
        while (is_pending(plain) || is_pending(spared)) && start.elapsed() < std::time::Duration::from_secs(20) {
            let _ = resolve(spared);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!waits_for(plain) && !waits_for(spared), "built");
        let _ = released(plain);
        let _ = released(spared);
    }

    #[test]
    fn the_put_off_line_says_what_became_of_the_put_off_dispatches() {
        let last = PutOffStats::default();
        assert_eq!(put_off_line(&last, &last), None);
        let idle = PutOffStats { builds: 5, ..last };
        assert_eq!(put_off_line(&idle, &last), None, "nothing put off: no line");
        let now = PutOffStats { calls: [12, 1, 9, 2, 1], needed: 7, needed_first: 5, built_here: 3, built_at_submit: 2, builds: 40, waits: (6, 2_500_000), segments: 24 };
        let before = PutOffStats { calls: [4, 0, 4, 0, 0], waits: (2, 500_000), ..Default::default() };
        let line = put_off_line(&now, &before).unwrap();
        assert!(line.contains("12 put off to the submit (8 in 2 s), 1 waited in the call"), "{line}");
        assert!(line.contains("at submits 9 recorded, 2 skipped (build failed or late), 1 dropped"), "{line}");
        assert!(line.contains("7 pipelines needed by a submit (5 built by a worker before its queue)"), "{line}");
        assert!(line.contains("submits waited 2000 ms for them in 2 s (4 waits; 2500 ms so far)"), "{line}");
        assert!(line.contains("3 builds of 40 on threads that waited for them (2 at a submit)"), "{line}");
        assert!(line.contains("at most 24 segments in a recording"), "{line}");
        assert_eq!(put_off_line(&now, &now), None, "nothing changed: no line");
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    /// The pipeline line's counts: lazy pipelines made (graphics among them), built at their
    /// first use or before it, with a huge program; a bind's wait is counted with its time, and
    /// one under a second is not logged.
    #[test]
    fn lazy_pipelines_and_the_waits_for_them_are_counted() {
        let before = stats();
        let (first_use, ahead) = (0x7E57_C001usize, 0x7E57_C002usize);
        submit_lazy(first_use, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            note_huge();
            0x6161
        }), 0);
        let key = 0x5EED_C002u64;
        used::earlier_run_used(key);
        submit_lazy(ahead, false, Box::new(|| 0x6262), key);
        let start = std::time::Instant::now();
        while is_pending(ahead) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(ahead), "built ahead with no bind");
        assert_eq!(resolve_wait_for(first_use, None), 0x6161);
        assert!(!entry(first_use).unwrap().warned.load(Ordering::Relaxed), "a short wait is not logged");
        let after = stats();
        assert!(after.made >= before.made + 2 && after.graphics > before.graphics, "{before:?} → {after:?}");
        assert!(after.built >= before.built + 2 && after.ahead > before.ahead && after.huge > before.huge, "{before:?} → {after:?}");
        assert!(after.waits.0 > before.waits.0 && after.waits.1 >= before.waits.1 + 5000, "{before:?} → {after:?}");
        released(first_use);
        released(ahead);
    }

    /// The line is written when a count changed, with the waits of its 2 s only.
    #[test]
    fn the_pipeline_line_comes_when_a_count_changed() {
        let last = Stats { made: 10, graphics: 8, built: 2, ahead: 1, huge: 0, warmed: 4, backlog: 6, waits: (3, 4_000_000) };
        assert_eq!(stats_line(&last, &last, 256), None);
        let now = Stats { built: 3, waits: (5, 6_500_000), ..last };
        let line = stats_line(&now, &last, 256).unwrap();
        assert!(line.contains("10 made (8 graphics), 3 built (1 before their first use, 0 with 256 KB of DXIL or more), 4 translated ahead, 6 in the build-ahead backlog"), "{line}");
        assert!(line.contains("waited 2500 ms for pipelines being built (2 waits in 2 s)"), "{line}");
    }

    /// A wait that passes a second is noted on its pipeline (and logged once for it).
    #[test]
    fn a_wait_past_a_second_is_noted_on_its_pipeline() {
        let standin = 0x7E57_C003usize;
        submit(standin, false, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(1100));
            0x6363
        }));
        let p = entry(standin).unwrap();
        assert_eq!(resolve_wait_for(standin, None), 0x6363);
        assert!(p.warned.load(Ordering::Relaxed));
        released(standin);
    }

    /// A build that takes 600 ms and gives `real`.
    fn slow(real: usize) -> Job {
        Box::new(move || {
            std::thread::sleep(std::time::Duration::from_millis(600));
            real
        })
    }

    /// The budget for graphics pipelines is a thread's for a second, not one for each pipeline:
    /// once it is spent the next pipeline is not waited for at all, it comes back with time
    /// (half of it in half a second) and never grows past itself. With a budget for each
    /// pipeline a title's thread stood still that long for every one of hundreds of pipelines
    /// it used for the first time.
    #[test]
    fn the_graphics_budget_is_a_threads_for_a_second() {
        use std::time::{Duration, Instant};
        std::thread::spawn(|| {
            let budget = Duration::from_millis(100);
            let t0 = Instant::now();
            assert_eq!(gfx_wait_left(budget, t0), budget, "all of it at first");
            gfx_wait_spent(Duration::from_millis(100));
            assert_eq!(gfx_wait_left(budget, t0), Duration::ZERO, "spent: the next pipeline is not waited for");
            let half = gfx_wait_left(budget, t0 + Duration::from_millis(500));
            assert!((half.as_secs_f64() - 0.050).abs() < 0.001, "half a second later half of it is back: {half:?}");
            assert_eq!(gfx_wait_left(budget, t0 + Duration::from_secs(5)), budget, "never more than the budget");
            gfx_wait_spent(Duration::from_millis(30));
            let left = gfx_wait_left(budget, t0 + Duration::from_secs(5));
            assert!((left.as_secs_f64() - 0.070).abs() < 0.001, "what was waited is taken off: {left:?}");
            gfx_wait_spent(Duration::from_secs(9));
            assert_eq!(gfx_wait_left(budget, t0 + Duration::from_secs(5)), Duration::ZERO, "a long wait leaves nothing, not a debt");
        })
        .join()
        .unwrap();
    }

    /// With a budget for graphics pipelines a draw waits no longer than it and later draws of
    /// that pipeline do not wait; a dispatch waits for its compute pipeline however long it takes.
    #[test]
    fn only_a_graphics_pipeline_is_left_after_its_budget() {
        use std::time::{Duration, Instant};
        let (gfx, cs) = (0x7E57_C020usize, 0x7E57_C021usize);
        submit(gfx, false, slow(0x6565));
        submit(cs, true, slow(0x6666));
        let budget = Some(Duration::from_millis(30));
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), gfx, "not built within the budget: the stand-in");
        assert!(t.elapsed() < Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), gfx);
        assert!(t.elapsed() < Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        assert_eq!(resolve_wait_budgets(cs, None, budget, None), 0x6666, "a compute pipeline is waited for");
        let start = Instant::now();
        while is_pending(gfx) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_budgets(gfx, None, budget, None), 0x6565, "built by a worker");
        let all = Some(Duration::from_secs(5));
        assert_eq!(budget_for(true, all, budget, None), all, "XCL_PSO_WAIT_MS still holds for compute");
        assert_eq!(budget_for(false, all, None, None), all);
        assert_eq!(budget_for(false, all, budget, None), budget);
        assert_eq!(budget_of(None), None);
        assert_eq!(budget_of(Some("0")), None, "0: wait until built");
        assert_eq!(budget_of(Some(" 250 ")), Some(Duration::from_millis(250)));
        assert_eq!(budget_of(Some("soon")), None);
        released(gfx);
        released(cs);
    }

    /// A build a bind gave up waiting for is no longer wanted (it goes back behind the gate of
    /// huge driver compiles); it was wanted while the bind waited.
    #[test]
    fn a_build_is_no_longer_wanted_once_its_wait_ran_out() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C022usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, false, Box::new(move || {
            started_tx.send(()).unwrap();
            let until = |want: bool| {
                let t = Instant::now();
                while build_is_wanted() != want && t.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                build_is_wanted() == want
            };
            if until(true) && until(false) { 0x6767 } else { 1 }
        }));
        started.recv().unwrap();
        assert_eq!(resolve_wait_budgets(standin, None, Some(Duration::from_millis(300)), None), standin);
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve(standin), 0x6767, "wanted while the bind waited, not after it gave up");
        released(standin);
    }

    /// With a budget of compute pipelines (`XCL_CS_PSO_WAIT_MS`) a dispatch waits no longer than
    /// it and later dispatches of that pipeline do not wait; the bind's thread does not build the
    /// pipeline, a worker does – also a lazy one nobody sent to the workers yet. Without it a
    /// compute pipeline is waited for as before.
    #[test]
    fn a_compute_pipeline_is_waited_for_no_longer_than_its_own_budget() {
        use std::time::{Duration, Instant};
        let standin = 0x7E57_C023usize;
        let (built_on, on) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = built_on.send(std::thread::current().id());
            let _ = finish.recv();
            0x6868
        }), 0);
        let cs = Some(Duration::from_millis(30));
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < Duration::from_millis(400), "waited {:?}", t.elapsed());
        let builder = on.recv_timeout(Duration::from_secs(10)).expect("a worker builds it");
        assert_ne!(builder, std::thread::current().id(), "the bind's thread does not build it");
        let t = Instant::now();
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), standin);
        assert!(t.elapsed() < Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        go.send(()).unwrap();
        let start = Instant::now();
        while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_budgets(standin, None, None, cs), 0x6868, "built by a worker");
        released(standin);
        let (all, gfx) = (Some(Duration::from_secs(5)), Some(Duration::from_millis(100)));
        assert_eq!(budget_for(true, None, None, None), None, "unset: a compute pipeline is waited for until it is built");
        assert_eq!(budget_for(true, None, gfx, None), None, "the graphics budget is not one of compute pipelines");
        assert_eq!(budget_for(true, all, gfx, None), all, "unset: XCL_PSO_WAIT_MS as before");
        assert_eq!(budget_for(true, all, gfx, cs), cs, "its own before XCL_PSO_WAIT_MS");
        assert_eq!(budget_for(true, None, None, cs), cs);
        assert_eq!(budget_for(false, all, None, cs), all, "not one of graphics pipelines");
        assert_eq!(budget_for(false, None, gfx, cs), gfx);
        assert_eq!(budget_for(false, None, None, cs), None);
    }

    /// A late compute program is logged once, not once for each of its pipelines.
    #[test]
    fn a_late_compute_program_is_logged_once() {
        assert!(first_late(0x5EED_C024));
        assert!(!first_late(0x5EED_C024), "another pipeline of the same program");
        assert!(first_late(0x5EED_C025));
    }

    /// A compute pipeline sent to the workers when made (`XCL_LAZY_PSO=0`) carries its console
    /// program as a lazy one does: a late one is logged by it (`resolve_within`).
    #[test]
    fn a_pipeline_sent_to_the_workers_when_made_carries_its_record() {
        let standin = 0x7E57_C026usize;
        let (go, finish) = mpsc::channel::<()>();
        submit_with(standin, true, Box::new(move || {
            let _ = finish.recv();
            0x6A6A
        }), 0x5EED_C026);
        assert_eq!(entry(standin).map(|p| p.record), Some(0x5EED_C026));
        go.send(()).unwrap();
        assert_eq!(resolve_wait_for(standin, None), 0x6A6A);
        released(standin);
    }

    /// `peek` names the real pipeline once built and, unlike `resolve`, neither notes a use nor
    /// sends a lazy one to the workers.
    #[test]
    fn a_peek_changes_nothing() {
        let standin = 0x7E57_C0DEusize;
        let p = pending(0x5EED_C0DE, false);
        PENDING.insert(standin, p.clone());
        assert_eq!(peek(standin), standin, "not built");
        assert!(!p.sent.load(Ordering::Acquire) && !p.used.load(Ordering::Relaxed));
        p.real.store(0x6A6B, Ordering::Release);
        assert_eq!(peek(standin), 0x6A6B);
        assert_eq!(peek(0x7E57_C0DF), 0x7E57_C0DF, "not async");
        let _ = PENDING.remove(standin);
    }

    /// A pipeline for the backlog (`warm`: with its translation on its own), never built.
    fn pending(key: u64, warm: bool) -> Arc<Pending> {
        Arc::new(Pending::new(Box::new(|| 0), false, false, key, warm.then(|| Box::new(|| {}) as Warm), 0))
    }

    /// A compute pipeline for the backlog, never built.
    fn pending_compute(key: u64) -> Arc<Pending> {
        Arc::new(Pending::new(Box::new(|| 0), false, true, key, None, 0))
    }

    /// A backlog worker's build nobody waits for builds the pieces of its dispatch chain itself:
    /// none of them goes to the workers whose builds binds wait for.
    #[test]
    fn a_build_ahead_keeps_its_pieces_on_its_own_thread() {
        let p = pending_compute(0x5EED_C1A1);
        p.demand.lock().unwrap().preload_thread = 1;
        let before = BUILDING.with(|b| b.replace(Some(p.clone())));
        let here = std::thread::current().id();
        let jobs: Vec<Job> = (0..6)
            .map(|_| {
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    usize::from(std::thread::current().id() == here)
                }) as Job
            })
            .collect();
        let on_here = build_pieces(jobs);
        BUILDING.with(|b| *b.borrow_mut() = before);
        assert_eq!(on_here, vec![1; 6], "every piece on the backlog worker's thread");
    }

    /// A piece whose build panics fails (0), on the caller's thread or a helper's, and the
    /// caller still gets every other piece.
    #[test]
    fn a_piece_that_panics_fails_and_the_others_are_built() {
        let jobs: Vec<Job> = (0..8usize)
            .map(|i| {
                Box::new(move || {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    if i % 2 == 1 {
                        panic!("piece {i}");
                    }
                    i + 1
                }) as Job
            })
            .collect();
        assert_eq!(build_pieces(jobs), [1, 0, 3, 0, 5, 0, 7, 0]);
    }

    /// Work from the backlog as its workers do (a translated one goes back for its build): the
    /// key and step of each piece, in order.
    fn drain(b: &mut Backlog, ahead: bool, builds: bool) -> Vec<(u64, &'static str)> {
        let mut done = Vec::new();
        while let Some((p, step)) = b.next(ahead, builds) {
            match step {
                Step::Warm(w) => {
                    w();
                    done.push((p.key, "warm"));
                    b.cold.push_back(p);
                }
                Step::Build { ahead, new_compute } => {
                    p.job.lock().unwrap().take();
                    done.push((p.key, if ahead { "build ahead" } else { "build" }));
                    b.built(new_compute);
                }
            }
        }
        done
    }

    /// The pipelines of earlier runs come first; then every other one is translated before the
    /// driver builds any of them; one a bind wants is left to the workers.
    #[test]
    fn the_backlog_builds_known_pipelines_first_and_translates_the_rest_before_building_it() {
        let mut b = Backlog::default();
        let wanted = pending(4, true);
        wanted.sent.store(true, Ordering::Release);
        b.push(pending(2, true), false);
        b.push(wanted, false);
        b.push(pending(5, false), false);
        b.push(pending(3, true), false);
        b.push(pending(1, false), true);
        assert_eq!(drain(&mut b, true, true), [(1, "build"), (2, "warm"), (3, "warm"), (5, "build ahead"), (2, "build ahead"), (3, "build ahead")]);
        assert_eq!(b.len(), 0);
    }

    /// Switched off or short of commit (`ahead` false), only the pipelines of earlier runs are
    /// built; past the cap (`builds` false) the others are translated but not built. They wait.
    #[test]
    fn nothing_is_built_ahead_when_switched_off_or_short_of_commit() {
        let mut b = Backlog::default();
        b.push(pending(1, true), false);
        b.push(pending(2, false), false);
        b.push(pending(3, false), true);
        assert_eq!(drain(&mut b, false, true), [(3, "build")]);
        assert_eq!(b.len(), 2, "the others wait");
        assert_eq!(drain(&mut b, true, false), [(1, "warm")]);
        assert_eq!(b.len(), 2);
        assert_eq!(drain(&mut b, true, true), [(2, "build ahead"), (1, "build ahead")]);
        assert_eq!(ahead_cap_of(None), 4096);
        assert_eq!(ahead_cap_of(Some(" 0 ")), 0, "XCL_PSO_AHEAD=0: off");
        assert_eq!(ahead_cap_of(Some("100")), 100);
        assert_eq!(ahead_cap_of(Some("lots")), 4096);
    }

    /// Compute pipelines no earlier run used have a lane of their own, after the known compute
    /// ones and before the known graphics ones; like every pipeline no earlier run used they are
    /// built only with room (`ahead`) and under the cap (`builds`), and one a bind wants is left
    /// to the workers.
    #[test]
    fn new_compute_pipelines_are_built_after_known_compute_and_before_known_graphics_ones() {
        let mut b = Backlog::default();
        let wanted = pending_compute(6);
        wanted.sent.store(true, Ordering::Release);
        b.push(pending(1, true), false);
        b.push(pending(2, false), true);
        b.push(pending_compute(3), false);
        b.push(wanted, false);
        b.push(pending_compute(4), true);
        b.push(pending_compute(5), false);
        assert_eq!(drain(&mut b, true, true), [(4, "build"), (3, "build ahead"), (5, "build ahead"), (2, "build"), (1, "warm"), (1, "build ahead")]);
        assert_eq!(b.len(), 0);
        b.push(pending_compute(3), false);
        b.push(pending(2, false), true);
        b.push(pending_compute(4), true);
        assert_eq!(drain(&mut b, false, true), [(4, "build"), (2, "build")]);
        assert_eq!(b.len(), 1, "short of commit: it waits");
        assert!(drain(&mut b, true, false).is_empty());
        assert_eq!(b.len(), 1, "past the cap: it waits");
        assert_eq!(drain(&mut b, true, true), [(3, "build ahead")]);
    }

    /// While a worker builds a compute pipeline no earlier run used, the other workers get the
    /// known graphics ones (and the rest), not the next new compute one.
    #[test]
    fn one_worker_at_a_time_builds_new_compute_pipelines() {
        let mut b = Backlog::default();
        b.push(pending_compute(3), false);
        b.push(pending_compute(4), false);
        b.push(pending(2, false), true);
        b.push(pending(1, false), false);
        let take = |b: &mut Backlog| b.next(true, true).map(|(p, step)| (p.key, matches!(step, Step::Build { new_compute: true, .. })));
        assert_eq!(take(&mut b), Some((3, true)));
        assert_eq!(take(&mut b), Some((2, false)), "a known graphics one while the new compute one is built");
        assert_eq!(take(&mut b), Some((1, false)));
        assert_eq!(take(&mut b), None, "the next new compute one waits for the one being built");
        b.built(false);
        assert_eq!(take(&mut b), None);
        b.built(true);
        assert_eq!(take(&mut b), Some((4, true)));
    }

    /// A pipeline binds go on without (`image_only`) goes to the backlog at its first bind, not to
    /// the workers builds a bind waits for go to, and is built in the lane of new compute ones:
    /// before them and one at a time with them, also short of commit and past the cap. Its place
    /// in a lane from when it was made is passed over.
    #[test]
    fn a_pipeline_binds_go_on_without_is_built_by_the_backlog_one_at_a_time() {
        let q = Preload { list: Mutex::new(Backlog::default()), cv: Condvar::new() };
        let spared = pending_compute(7);
        {
            let mut b = q.list.lock().unwrap();
            b.push(pending_compute(3), false);
            b.push(spared.clone(), false);
            b.push(pending(2, false), true);
        }
        // (as `resolve_spared` does)
        spared.sent.store(true, Ordering::Release);
        send_spared(spared, Some(&q));
        let mut b = q.list.lock().unwrap();
        let mut take = |ahead: bool| {
            b.next(ahead, ahead).map(|(p, step)| {
                p.job.lock().unwrap().take();
                let Step::Build { ahead, new_compute } = step else { panic!("{} was not built", p.key) };
                (p.key, ahead, new_compute)
            })
        };
        assert_eq!(take(false), Some((7, false, true)), "short of commit and past the cap it is built, in the new compute lane");
        assert_eq!(take(true), Some((2, false, false)), "a known graphics one while it is built");
        assert_eq!(take(true), None, "the next new compute one waits for it");
        b.built(true);
        assert_eq!(b.next(true, true).map(|(p, _)| p.key), Some(3));
        b.built(true);
        assert!(b.next(true, true).is_none(), "its place from when it was made is passed over");
        assert_eq!(b.len(), 0);
    }

    /// Bound under a wait budget, a pipeline binds go on without is built by a backlog worker,
    /// not by the workers builds a bind waits for go to.
    #[test]
    fn a_pipeline_binds_go_on_without_is_not_built_by_the_workers_binds_wait_for() {
        let standin = 0x7E57_B0A3usize;
        let (tx, rx) = mpsc::channel();
        submit_lazy(standin, true, Box::new(move || {
            let _ = tx.send(std::thread::current().name().map(str::to_owned));
            0x5C5C
        }), 0);
        image_only(standin);
        assert_eq!(resolve_wait_for(standin, Some(std::time::Duration::from_millis(30))), standin, "still being built: the stand-in");
        let on = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a worker builds it");
        assert!(on.as_deref().is_some_and(|n| n.starts_with("xcl-pso-pre-")), "built on {on:?}");
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, None), 0x5C5C);
        let _ = released(standin);
    }

    /// A lazy compute pipeline of shaders no run used yet is built with no bind – when commit is
    /// free.
    #[test]
    fn a_lazy_compute_pipeline_of_new_shaders_is_built_ahead_of_its_first_use() {
        let standin = 0x7E57_C011usize;
        let room = ahead_room();
        submit_lazy(standin, true, Box::new(|| 0x6969), 0x5EED_C011);
        let start = std::time::Instant::now();
        let limit = std::time::Duration::from_secs(if room { 10 } else { 1 });
        while is_pending(standin) && start.elapsed() < limit {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if room {
            assert!(!is_pending(standin), "built with no bind");
            assert_eq!(resolve(standin), 0x6969);
        } else {
            assert!(is_pending(standin), "short of commit: not built ahead");
        }
        released(standin);
    }

    /// The names of the backlog's workers alive now (`spawn_preload_worker`).
    fn preload_workers() -> Vec<String> {
        use windows_sys::Win32::Foundation::{CloseHandle, LocalFree, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32};
        use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
        let mut names = Vec::new();
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            assert_ne!(snap, INVALID_HANDLE_VALUE, "no thread snapshot");
            let mut te: THREADENTRY32 = std::mem::zeroed();
            te.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
            let mut ok = Thread32First(snap, &mut te) != 0;
            while ok {
                let h = if te.th32OwnerProcessID == GetCurrentProcessId() { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, te.th32ThreadID) } else { std::ptr::null_mut() };
                if !h.is_null() {
                    let mut p: *mut u16 = std::ptr::null_mut();
                    if GetThreadDescription(h, &mut p) >= 0 && !p.is_null() {
                        let n = (0..).take_while(|&i| *p.add(i) != 0).count();
                        let name = String::from_utf16_lossy(std::slice::from_raw_parts(p, n));
                        LocalFree(p.cast());
                        if name.starts_with("xcl-pso-pre-") {
                            names.push(name);
                        }
                    }
                    CloseHandle(h);
                }
                ok = Thread32Next(snap, &mut te) != 0;
            }
            CloseHandle(snap);
        }
        names
    }

    /// A giant build ahead on a backlog worker waits for its pipeline's first use
    /// (`real::huge_compile_turn`) and does not stop the lane of new compute pipelines meanwhile:
    /// the next one is built while it waits, by the worker that took the waiting one's place
    /// (`build_parks`). Once a bind waits for it, it gets the giant slot and is built; its thread
    /// then ends without giving the lane back a second time (it holds another worker's build), and
    /// as many workers are left as before. Without that a pipeline the title never uses held the
    /// lane for good.
    #[test]
    fn a_parked_giant_build_ahead_leaves_the_lane_of_new_compute_pipelines_to_the_next() {
        use std::time::{Duration, Instant};
        let until = |done: &dyn Fn() -> bool, limit: Duration| {
            let start = Instant::now();
            while !done() && start.elapsed() < limit {
                std::thread::sleep(Duration::from_millis(5));
            }
            done()
        };
        if preload().is_none() || !ahead_room() {
            crate::gpu::recomp::runner::skip_test("no build ahead (XCL_PSO_PRELOAD=0, XCL_PSO_AHEAD=0 or short of commit)");
            return;
        }
        // (as `preload` starts them; a thread names itself once it runs)
        let workers = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        assert!(until(&|| preload_workers().len() == workers, Duration::from_secs(10)), "the backlog's workers: {:?}", preload_workers());
        let (giant, next, holder, last) = (0x7E57_C030usize, 0x7E57_C031usize, 0x7E57_C032usize, 0x7E57_C033usize);
        // the giant build: its worker's name, then its turn at the giant slot
        let (on_tx, on) = mpsc::channel();
        let (through_tx, through) = mpsc::channel();
        submit_lazy(giant, true, Box::new(move || {
            let _ = on_tx.send(std::thread::current().name().map(str::to_owned));
            let slot = super::super::real::huge_compile_turn(super::super::real::GIANT_DRIVER_COMPILE);
            let _ = through_tx.send(());
            drop(slot);
            0x6B6B
        }), 0x5EED_C030);
        let parked = on.recv_timeout(Duration::from_secs(10)).expect("a backlog worker builds it ahead").unwrap_or_default();
        assert!(parked.starts_with("xcl-pso-pre-"), "built on {parked:?}");
        submit_lazy(next, true, Box::new(|| 0x6C6C), 0x5EED_C031);
        assert!(until(&|| !is_pending(next), Duration::from_secs(10)), "the next new compute pipeline is not built while the giant one waits");
        assert!(is_pending(giant), "nobody used the giant one: it is not built");
        assert!(through.try_recv().is_err(), "a giant build ahead took its turn");
        assert!(until(&|| preload_workers().len() == workers + 1, Duration::from_secs(10)), "another worker took the waiting one's place: {:?}", preload_workers());
        // a new compute build the lane holds when the giant build ends
        let (held_tx, held) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(holder, true, Box::new(move || {
            let _ = held_tx.send(());
            let _ = finish.recv_timeout(Duration::from_secs(20));
            0x6D6D
        }), 0x5EED_C032);
        held.recv_timeout(Duration::from_secs(10)).expect("the lane's next build starts");
        assert_eq!(resolve_wait_for(giant, None), 0x6B6B, "wanted, the giant build is made");
        assert!(through.try_recv().is_ok(), "it went through the giant slot");
        assert!(until(&|| !preload_workers().contains(&parked), Duration::from_secs(10)), "the waiting worker's thread did not end");
        assert_eq!(preload_workers().len(), workers, "as many workers as before: {:?}", preload_workers());
        // the lane is still the holder's: the giant build's end did not give it back again
        submit_lazy(last, true, Box::new(|| 0x6E6E), 0x5EED_C033);
        std::thread::sleep(Duration::from_millis(300));
        assert!(is_pending(last), "a second new compute build went while one was out");
        go.send(()).unwrap();
        assert!(until(&|| !is_pending(last), Duration::from_secs(10)), "the lane went back after the holder's build");
        assert_eq!(resolve(next), 0x6C6C);
        assert_eq!(resolve(holder), 0x6D6D);
        assert_eq!(resolve(last), 0x6E6E);
        for standin in [giant, next, holder, last] {
            released(standin);
        }
    }

    /// A lazy graphics pipeline of shaders no run used yet is translated, then built, with no
    /// bind – when commit is free.
    #[test]
    fn a_lazy_graphics_pipeline_is_translated_then_built_ahead_of_its_first_use() {
        let standin = 0x7E57_C010usize;
        let warmed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let w = warmed.clone();
        let room = ahead_room();
        submit_lazy_with(standin, false, Box::new(move || if w.load(Ordering::Acquire) { 0x6464 } else { 1 }), 0x5EED_C010, Some(Box::new(move || warmed.store(true, Ordering::Release))), 0);
        let start = std::time::Instant::now();
        let limit = std::time::Duration::from_secs(if room { 10 } else { 1 });
        while is_pending(standin) && start.elapsed() < limit {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        if room {
            assert!(!is_pending(standin), "built with no bind");
            assert_eq!(resolve(standin), 0x6464, "translated before the driver built it");
        } else {
            assert!(is_pending(standin), "short of commit: not built ahead");
        }
        released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; a compute one of
    /// other shaders is not used before its first bind (built ahead or not:
    /// `a_lazy_compute_pipeline_of_new_shaders_is_built_ahead_of_its_first_use`), and that use is
    /// kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, true, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(!used::noted(unknown_key), "nobody used it and no earlier run did");
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }

    /// What a pipeline's first use tells (a graphics one's description: `gfx_records`) is told
    /// once, at that use, not when it is made or built.
    #[test]
    fn a_first_use_is_told_once() {
        let (standin, key) = (0x7E57_7011usize, 0x5EED_0011u64);
        let told = Arc::new(AtomicUsize::new(0));
        submit_lazy(standin, true, Box::new(|| 0x4331), key);
        let t = told.clone();
        on_first_use(standin, Box::new(move || { t.fetch_add(1, Ordering::SeqCst); }));
        assert_eq!(told.load(Ordering::SeqCst), 0, "not at its making");
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1);
        assert_eq!(resolve(standin), 0x4331);
        assert_eq!(resolve_wait(standin), 0x4331);
        assert_eq!(told.load(Ordering::SeqCst), 1, "once");
        on_first_use(0x7E57_7012, Box::new(|| unreachable!("no such pipeline")));
        released(standin);
    }

    static WORKER_SATURATION_LOCK: Mutex<()> = Mutex::new(());

    /// A queued normal worker job whose stand-in was released before a worker starts it is
    /// skipped as orphaned instead of entering the compiler.
    #[test]
    fn a_released_queued_pipeline_is_not_built_by_normal_workers() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_9720usize + k;
            let started_tx = started_tx.clone();
            let gate = gate.clone();
            submit(standin, false, Box::new(move || {
                let _ = started_tx.send(());
                let mut open = gate.0.lock().unwrap();
                while !*open {
                    open = gate.1.wait(open).unwrap();
                }
                0x9720 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }

        let target = 0x7E57_97FFusize;
        let ran = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let ran_job = ran.clone();
        let drop_flag = DropFlag(dropped.clone());
        submit(target, false, Box::new(move || {
            let _keep = &drop_flag;
            ran_job.store(true, Ordering::Release);
            0x9721
        }));
        assert_eq!(released(target), 0);

        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !dropped.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(dropped.load(Ordering::Acquire), "queued orphan job was never drained");
        assert!(!ran.load(Ordering::Acquire), "released job reached the compiler");

        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }

    #[test]
    fn pc_call_crossing_frame_boundary_does_not_charge_prior_frame_time() {
        let mut calls = PcCalls::new();
        let t0 = std::time::Instant::now();
        let ms = std::time::Duration::from_millis;
        calls.begin(1, 7, t0);
        // Frame 8 shown after 290 ms; call 1 ends 10 ms later -> only 10 ms counts on frame 8.
        let t_frame8 = t0 + PC_CALL_HOLDS_AFTER + ms(40);
        assert!(!calls.holds(8, t_frame8), "call from frame 7 must not immediately hold frame 8");
        calls.end(1, 8, t_frame8 + ms(10));
        let t_next = t_frame8 + ms(11);
        calls.begin(2, 8, t_next);
        assert!(!calls.holds(8, t_next), "pre-frame duration leaked into frame 8's budget");
        // A call genuinely stuck >= PC_CALL_HOLDS_AFTER after frame 8 still holds the title.
        assert!(!calls.holds(8, t_next + PC_CALL_HOLDS_AFTER - ms(20)));
        assert!(calls.holds(8, t_next + PC_CALL_HOLDS_AFTER));
        calls.end(2, 8, t_next + PC_CALL_HOLDS_AFTER);

        // Also when frame 9 is first noticed inside end() itself, pre-frame time is clamped to 0.
        let t_f9 = t_next + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(3, 8, t_f9);
        let t_f9_end = t_f9 + PC_CALL_HOLDS_AFTER + ms(50);
        calls.end(3, 9, t_f9_end);
        calls.begin(4, 9, t_f9_end + ms(1));
        assert!(!calls.holds(9, t_f9_end + ms(1)), "end() on new frame must not charge prior-frame duration");
        calls.end(4, 9, t_f9_end + ms(1) + PC_CALL_HOLDS_AFTER);

        // A stale lower frame number must never rewind self.frame or wipe frame 9's spent budget.
        let t_stale = t_f9_end + PC_CALL_HOLDS_AFTER + ms(10);
        calls.begin(5, 8, t_stale);
        assert!(calls.holds(9, t_stale), "stale lower frame rewound frame 9 or cleared its spent budget");
        calls.end(5, 9, t_stale);
    }

    #[test]
    fn released_queued_pipeline_drops_job_immediately_before_workers_drain() {
        let _sat = WORKER_SATURATION_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::AtomicBool;
        use std::time::{Duration, Instant};

        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        struct OpenGate(Arc<(Mutex<bool>, Condvar)>);
        impl Drop for OpenGate {
            fn drop(&mut self) {
                *self.0 .0.lock().unwrap() = true;
                self.0 .1.notify_all();
            }
        }

        // Saturate normal workers and verify released() drops the queued job immediately.
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_normal = OpenGate(gate.clone());
        let mut blockers = Vec::with_capacity(n);
        for k in 0..n {
            let standin = 0x7E57_A100usize + k;
            let (tx, g) = (started_tx.clone(), gate.clone());
            submit(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA100 + k
            }));
            blockers.push(standin);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(10)).expect("normal workers saturated");
        }
        let target_n = 0x7E57_A1FFusize;
        let (ran_n, dropped_n) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_n) = (ran_n.clone(), DropFlag(dropped_n.clone()));
        submit(target_n, false, Box::new(move || {
            let _keep = &flag_n;
            ran_job.store(true, Ordering::Release);
            0xA1FF
        }));
        assert_eq!(released(target_n), 0);
        assert!(dropped_n.load(Ordering::Acquire), "normal queued job was not dropped immediately on released()");
        drop(open_normal);
        assert!(!ran_n.load(Ordering::Acquire));
        for standin in blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }

        // Saturate preload workers and verify released() drops the preloaded job immediately.
        let m = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        let (p_tx, p_rx) = mpsc::channel();
        let p_gate = Arc::new((Mutex::new(false), Condvar::new()));
        let open_pre = OpenGate(p_gate.clone());
        let mut p_blockers = Vec::with_capacity(m);
        for k in 0..m {
            let standin = 0x7E57_A200usize + k;
            let key = 0x5EED_A200u64 + k as u64;
            used::earlier_run_used(key);
            let (tx, g) = (p_tx.clone(), p_gate.clone());
            submit_lazy(standin, false, Box::new(move || {
                let _ = tx.send(());
                let mut open = g.0.lock().unwrap();
                while !*open {
                    open = g.1.wait(open).unwrap();
                }
                0xA200 + k
            }), key);
            p_blockers.push(standin);
        }
        for _ in 0..m {
            p_rx.recv_timeout(Duration::from_secs(10)).expect("preload workers saturated");
        }
        let target_p = 0x7E57_A2FFusize;
        let key_p = 0x5EED_A2FFu64;
        used::earlier_run_used(key_p);
        let (ran_p, dropped_p) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        let (ran_job, flag_p) = (ran_p.clone(), DropFlag(dropped_p.clone()));
        submit_lazy(target_p, false, Box::new(move || {
            let _keep = &flag_p;
            ran_job.store(true, Ordering::Release);
            0xA2FF
        }), key_p);
        assert_eq!(released(target_p), 0);
        assert!(dropped_p.load(Ordering::Acquire), "preload queued job was not dropped immediately on released()");
        drop(open_pre);
        assert!(!ran_p.load(Ordering::Acquire));
        for standin in p_blockers {
            let start = Instant::now();
            while is_pending(standin) && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(2));
            }
            let _ = released(standin);
        }
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.get(pso)
}

impl Pending {
    /// What a bind of the stand-in `pso` gives the driver while the pipeline is not built.
    fn stand(&self, pso: usize) -> usize {
        match self.driver.load(Ordering::Acquire) {
            0 => pso,
            d => d,
        }
    }
}

/// Stand-in `standin` is no driver object (`standin::shell`): binds get `driver` in its place.
/// Told before the title has the stand-in.
pub fn shared(standin: usize, driver: usize) {
    if let Some(p) = entry(standin) {
        p.driver.store(driver, Ordering::Release);
    }
}

/// The pipeline to use for `pso` now: the real one once built, else the stand-in's driver
/// pipeline (`pso` itself when it is one).
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers (one binds do not wait for to the
            // backlog's lane, as `resolve_spared` sends it: a bind no longer waits for a compute
            // pipeline, its dispatch does)
            let stand = p.stand(pso);
            if !p.sent.swap(true, Ordering::AcqRel) {
                if p.spared.load(Ordering::Acquire) {
                    send_spared(p, preload());
                } else {
                    enqueue(p);
                }
            }
            stand
        }
        r => r,
    }
}

/// `resolve` with no side effects (for a report): the real pipeline behind `pso` once built,
/// else `pso` itself – no use noted, nothing sent to the workers.
pub(super) fn peek(pso: usize) -> usize {
    entry(pso).map(|p| p.real.load(Ordering::Acquire)).filter(|&r| r != 0).unwrap_or(pso)
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_budgets(pso, wait_budget(), gfx_wait_budget(), cs_wait_budget())
}

/// `resolve_wait` with the budget of every pipeline (`XCL_PSO_WAIT_MS`), the one of graphics
/// pipelines (`XCL_GFX_PSO_WAIT_MS`) and the one of compute pipelines (`XCL_CS_PSO_WAIT_MS`).
fn resolve_wait_budgets(pso: usize, all: Option<std::time::Duration>, gfx: Option<std::time::Duration>, cs: Option<std::time::Duration>) -> usize {
    let Some(compute) = entry(pso).map(|p| p.compute) else { return pso };
    match gfx.filter(|_| !compute) {
        // the graphics budget is a thread's for a second, not one for each pipeline
        Some(budget) => {
            let start = std::time::Instant::now();
            let r = resolve_wait_for(pso, Some(gfx_wait_left(budget, start)));
            gfx_wait_spent(start.elapsed());
            r
        }
        None => resolve_wait_for(pso, budget_for(compute, all, gfx, cs)),
    }
}

thread_local! {
    /// What this thread may still wait for graphics pipelines (`gfx_wait_left`): the
    /// milliseconds left, and when they were last counted.
    static GFX_WAIT: std::cell::Cell<(f64, Option<std::time::Instant>)> = const { std::cell::Cell::new((0.0, None)) };
}

/// How long this thread may wait for a graphics pipeline at `now`, with `budget` to spend in a
/// second: what is left of it (it fills again at `budget` a second, to `budget` at most).
///
/// The budget is not one for each pipeline: the first frames of a title's world bind hundreds
/// of pipelines never used before, one draw after another on one thread. With 500 ms for each,
/// a retail title's thread stood still 500 ms more than 500 times in a row on a cold cache and
/// showed no frame for minutes, although every one of those pipelines was being built
/// meanwhile. With a share of the thread's time instead, the wait is spent on the first ones
/// (and on any that is built within milliseconds: a pipeline made from a kept blob), the
/// thread goes on, and the rest draw once they are built.
fn gfx_wait_left(budget: std::time::Duration, now: std::time::Instant) -> std::time::Duration {
    GFX_WAIT.with(|w| {
        let (left, at) = w.get();
        let full = budget.as_secs_f64() * 1000.0;
        let left = match at {
            None => full,
            Some(at) => (left + now.saturating_duration_since(at).as_secs_f64() * full).min(full),
        };
        w.set((left, Some(now)));
        std::time::Duration::from_secs_f64(left / 1000.0)
    })
}

/// This thread waited `waited` for a graphics pipeline (`gfx_wait_left`).
fn gfx_wait_spent(waited: std::time::Duration) {
    GFX_WAIT.with(|w| {
        let (left, at) = w.get();
        w.set(((left - waited.as_secs_f64() * 1000.0).max(0.0), at));
    })
}

/// How long a bind waits for a pipeline of this kind: a compute one by `cs` if there is one, else
/// by `all` (a skipped dispatch leaves out data for later work: `cs_wait_budget`), a graphics one
/// by `gfx` if there is one.
fn budget_for(compute: bool, all: Option<std::time::Duration>, gfx: Option<std::time::Duration>, cs: Option<std::time::Duration>) -> Option<std::time::Duration> {
    if compute { cs.or(all) } else { gfx.or(all) }
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    if p.spared.load(Ordering::Acquire) {
        return resolve_spared(&p, pso);
    }
    if let Some(budget) = budget {
        return resolve_within(&p, pso, budget);
    }
    if *p.done.lock().unwrap() {
        return built_or(&p, pso);
    }
    // (the title's thread is held from here until the pipeline is built)
    let mut wait = Wait::new(&p);
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        BUILT_HERE.fetch_add(1, Ordering::Relaxed);
        if is_at_submit() {
            BUILT_AT_SUBMIT.fetch_add(1, Ordering::Relaxed);
        }
        complete(&p, run_job(&p, job));
    }
    p.want();
    wait.until_done(None);
    built_or(&p, pso)
}

/// `p`'s real pipeline once built, else `pso` (its stand-in).
fn built_or(p: &Pending, pso: usize) -> usize {
    match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    }
}

/// A wait of one of the title's threads for a pipeline still being built: the title is held
/// (`title_held`), the time is counted (`api_cost::Call::PipelineWait`, the pipeline line), and
/// the first wait for a pipeline that passes 1 s is logged with the thread and the pipeline's
/// kind – a draw or dispatch of the title stands still meanwhile.
struct Wait<'a> {
    p: &'a Pending,
    start: std::time::Instant,
    /// this wait logged that it passed 1 s
    logged: bool,
    _held: Held,
    _cost: crate::hle::api_cost::Timer,
}

/// A wait this long is logged (once a pipeline).
const LONG_WAIT: std::time::Duration = std::time::Duration::from_secs(1);

impl<'a> Wait<'a> {
    fn new(p: &'a Pending) -> Wait<'a> {
        Wait { p, start: std::time::Instant::now(), logged: false, _held: Held::new(), _cost: crate::hle::api_cost::time(crate::hle::api_cost::Call::PipelineWait) }
    }

    fn kind(&self) -> &'static str {
        if self.p.compute { "compute" } else { "graphics" }
    }

    /// Until the pipeline is built (true) or `budget` from the start of the wait is over
    /// (false; None: no budget).
    fn until_done(&mut self, budget: Option<std::time::Duration>) -> bool {
        loop {
            let waited = self.start.elapsed();
            if waited >= LONG_WAIT && !self.p.warned.swap(true, Ordering::Relaxed) {
                self.logged = true;
                crate::xlog_warn!(
                    "D3D12.X: thread {} has waited {} ms for a {} pipeline still being built (its draw or dispatch stands still)",
                    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
                    waited.as_millis(),
                    self.kind()
                );
            }
            let mut step = budget.map(|b| b.saturating_sub(waited));
            if !self.p.warned.load(Ordering::Relaxed) {
                let to_long = LONG_WAIT.saturating_sub(waited);
                step = Some(step.map_or(to_long, |s| s.min(to_long)));
            }
            let done = self.p.done.lock().unwrap();
            let done = match step {
                None => self.p.cv.wait_while(done, |d| !*d).unwrap(),
                Some(s) => self.p.cv.wait_timeout_while(done, s, |d| !*d).unwrap().0,
            };
            if *done {
                return true;
            }
            drop(done);
            if budget.is_some_and(|b| self.start.elapsed() >= b) {
                return false;
            }
        }
    }
}

impl Drop for Wait<'_> {
    fn drop(&mut self) {
        let waited = self.start.elapsed();
        if is_at_submit() {
            SUBMIT_WAITS.fetch_add(1, Ordering::Relaxed);
            SUBMIT_WAIT_US.fetch_add(waited.as_micros() as u64, Ordering::Relaxed);
        }
        // (a bind that built the pipeline itself did not wait on `until_done`)
        if self.logged || (waited >= LONG_WAIT && !self.p.warned.swap(true, Ordering::Relaxed)) {
            crate::xlog_warn!(
                "D3D12.X: thread {} waited {} ms for a {} pipeline being built",
                unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
                waited.as_millis(),
                self.kind()
            );
        }
    }
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_PSO_WAIT_MS").ok().as_deref()))
}

/// How long a thread's draws wait in a second for graphics pipelines still being built
/// (`XCL_GFX_PSO_WAIT_MS`; unset or 0: as `wait_budget`, for each pipeline; `gfx_wait_left` has
/// why it is a share of the thread's time). A draw past it goes on with the stand-in, which draws
/// nothing, and so do the later draws of that pipeline until it is built. On a retail title the
/// draws of its world waited 7 – 15 s each for pipelines built one after another: no frame for
/// 156 s. Off by default: on a cold cache a logo and UI text drawn once and kept were missing
/// when draws did not wait.
fn gfx_wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_GFX_PSO_WAIT_MS").ok().as_deref()))
}

/// How long a bind waits for a compute pipeline still being built (`XCL_CS_PSO_WAIT_MS`; unset
/// or 0: as `wait_budget`). With it the bind's thread never builds the pipeline itself: the build
/// goes to the workers (`resolve_within`; a huge driver compile also waits for free commit at the
/// gate once nobody waits), and past the budget the pipeline's dispatches are skipped until it is built – what
/// they compute is missing or old meanwhile, the counts of later ExecuteIndirects among it
/// (`indirect_guard` zeroes only counts out of range). A retail title's thread spent 57 s in the
/// driver's compiler on one program no earlier run had used, and another of its threads waited
/// for it all that time: no frame for 57 s.
/// A compute pipeline's dispatches go on without it while it is built (`XCL_CS_PSO_WAIT_MS`):
/// what builds one on a title's thread hands it to a thread of its own instead
/// (`real::dispatch_chain::whole_ready`).
pub(super) fn cs_goes_on() -> bool {
    cs_wait_budget().is_some()
}

fn cs_wait_budget() -> Option<std::time::Duration> {
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| budget_of(std::env::var("XCL_CS_PSO_WAIT_MS").ok().as_deref()))
}

/// A budget in milliseconds from the environment (unset, 0 or not a number: none).
fn budget_of(v: Option<&str>) -> Option<std::time::Duration> {
    v.and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis)
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// draws and dispatches are skipped) until a worker has built it. Its build is no longer wanted
/// (`Pending::unwant`), but it was used: it goes before the builds ahead of first use.
fn resolve_within(p: &Arc<Pending>, pso: usize, budget: std::time::Duration) -> usize {
    let built = |p: &Pending| match p.real.load(Ordering::Acquire) {
        0 => p.stand(pso),
        r => r,
    };
    if *p.done.lock().unwrap() {
        return built(p);
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
    }
    if p.late.load(Ordering::Acquire) {
        // (a submit that counted itself ahead, `want`, before another wait made it late: it
        // waits no more either)
        p.unwant_ahead();
        return p.stand(pso);
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let mut wait = Wait::new(p);
    p.want();
    if wait.until_done(Some(budget)) {
        return built(p);
    }
    drop(wait);
    p.late.store(true, Ordering::Release);
    p.unwant();
    p.unwant_ahead();
    if p.compute && p.record != 0 && cs_wait_budget().is_some() {
        if first_late(p.record) {
            crate::xlog_warn!("D3D12.X: compute pipeline of record {:016x} not built after {} ms: its dispatches are skipped until it is", p.record, budget.as_millis());
        }
        return p.stand(pso);
    }
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    p.stand(pso)
}

/// The first late compute pipeline of console program `record` (`resolve_within`): its line is
/// written once a program, which a title makes with several root signatures.
fn first_late(record: u64) -> bool {
    static SEEN: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
    SEEN.lock().unwrap().get_or_insert_with(Default::default).insert(record)
}

/// `XCL_PSO_IMAGE_SKIP=1`: a bind does not wait for a big compute program that writes nothing
/// but images (`gpu::effects::big_and_image_only`); its dispatches are left out until a worker
/// has built it. Off by default: what such a program draws is missing for that long.
///
/// A retail title's world start binds one ray-tracing resolve program per material (73 – 118
/// KiB of console code, about 1 MB of DXIL each): an RDNA2 driver takes 22 – 172 s for each,
/// one bind at a time, and the title stands still for all of them. `XCL_PSO_WAIT_MS` goes on without every late
/// program, also the culling passes whose counts an `ExecuteIndirect` reads next – a GPU hang.
pub fn image_skip() -> bool {
    static S: OnceLock<bool> = OnceLock::new();
    *S.get_or_init(|| std::env::var("XCL_PSO_IMAGE_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// The compute pipeline behind stand-in `standin` writes nothing but images: binds go on
/// without it while it is built (`resolve_spared`). Told before the title has the stand-in.
pub fn image_only(standin: usize) {
    if let Some(p) = entry(standin) {
        p.spared.store(true, Ordering::Release);
    }
}

/// `resolve_wait` for a pipeline binds do not wait for (`image_only`): the real one once built,
/// else the stand-in, with the build sent to the workers. Nobody waits, so the build is not
/// `wanted`: a big one takes its turn among the driver's compiles and waits for free commit.
fn resolve_spared(p: &Arc<Pending>, pso: usize) -> usize {
    match p.real.load(Ordering::Acquire) {
        0 => {
            if !p.sent.swap(true, Ordering::AcqRel) {
                send_spared(p.clone(), preload());
            }
            if !*p.done.lock().unwrap() && !p.late.swap(true, Ordering::AcqRel) {
                static N: AtomicUsize = AtomicUsize::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                if n < 16 || n.is_power_of_two() {
                    crate::xlog_info!("D3D12.X: a compute program that writes only images is still being built: its dispatches are left out until it is ({} so far)", n + 1);
                }
            }
            p.stand(pso)
        }
        r => r,
    }
}

/// The build of a pipeline binds do not wait for (`image_only`) goes to the backlog `q`, in the
/// lane of new compute pipelines (`Backlog::spared`: one at a time), not to the workers builds a
/// bind waits for go to (`queue`): there each of them, unwanted, waited for the one unwanted
/// slot of huge driver compiles, and with more of them than workers a build a bind waited for
/// stood in the queue behind them. Without the backlog (`XCL_PSO_PRELOAD=0`): to those workers.
fn send_spared(p: Arc<Pending>, q: Option<&Preload>) {
    let Some(q) = q else {
        enqueue(p);
        return;
    };
    let mut b = q.list.lock().unwrap();
    b.spared.push_back(p);
    BACKLOG.store(b.len(), Ordering::Relaxed);
    q.cv.notify_one();
}

/// What stand-in `pso` is (being) built from, for a diagnostic line (`dred::mark_pipeline`: a
/// pipeline bound with no label of its own). None: not a stand-in.
pub fn describe(pso: usize) -> Option<String> {
    let p = entry(pso)?;
    let state = if p.real.load(Ordering::Acquire) != 0 { "" } else { ", not built yet" };
    Some(match (p.compute, p.record) {
        (true, 0) => format!("a compute pipeline {pso:#x}{state}"),
        (true, record) => format!("cs {record:016x}{state}"),
        (false, _) => format!("a graphics pipeline {pso:#x} (shaders {:016x}){state}", p.key),
    })
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.remove(pso) else { return 0 };
    {
        let done = p.done.lock().unwrap();
        if *done {
            return p.real.swap(0, Ordering::AcqRel);
        }
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
    }
    drop_unstarted_job(&p);
    0
}

/// Whether a draw or dispatch with `pso` now has the pipeline it stands for: not a stand-in, or
/// one whose build succeeded (`put_off_line`'s skipped ones are those that had not).
pub(super) fn has_real(pso: usize) -> bool {
    entry(pso).is_none_or(|p| p.real.load(Ordering::Acquire) != 0)
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// `pso` is a stand-in whose real pipeline is being built or was built, and `resolved`, what a
/// `resolve_wait` gave for it, is the stand-in (`pso` itself, or the driver pipeline a shared
/// stand-in binds: `shared`): the bind has the stand-in, also when the build finished since (a
/// failed build's stand-in is what draws for good).
pub fn gave_standin(pso: usize, resolved: usize) -> bool {
    entry(pso).is_some_and(|p| resolved == p.stand(pso) && (!*p.done.lock().unwrap() || p.real.load(Ordering::Acquire) != 0))
}

/// Whether a dispatch with `pso` would wait for its build now (`resolve_wait`): a pipeline still
/// being built that binds do not go on without – not one whose dispatches are left out while it
/// is built (`image_only`) or that a bind already waited for past its budget (`late`). Those go
/// on as they did, in the call, and their builds keep their place (not before the workers'
/// queue: `need_by_submit`).
pub fn waits_for(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !p.spared.load(Ordering::Acquire) && !p.late.load(Ordering::Acquire) && !*p.done.lock().unwrap())
}

/// A submit is about to wait for `pso` (one of its next put-off calls): it counts among the
/// build's waiters (`Pending::want`) once, however often the submit says so, and leaves with
/// the wait for it when that runs out of its budget (`resolve_within`). Not for a pipeline the
/// submit will not wait for (late, or spared): nothing would take that count back, and its build
/// would stay wanted – ahead of the commit gate – until it is built.
pub fn want(pso: usize) {
    let Some(p) = entry(pso) else { return };
    // (under the lock: a wait that makes it late takes `ahead` back after it, `resolve_within`)
    let mut d = p.demand.lock().unwrap();
    if p.late.load(Ordering::Acquire) || p.spared.load(Ordering::Acquire) || std::mem::replace(&mut d.ahead, true) {
        return;
    }
    Pending::add_waiter(&mut d);
}

/// A call with `pso` was put off to the submit: its build goes before the workers' queue.
pub fn need_by_submit(pso: usize) {
    let Some(p) = entry(pso) else { return };
    // (a lazy one is sent here: `resolve` does not send it again)
    p.sent.store(true, Ordering::Release);
    if !std::mem::replace(&mut p.demand.lock().unwrap().by_submit, true) {
        NEEDED_MADE.fetch_add(1, Ordering::Relaxed);
        NEEDED.lock().unwrap().push_back(p);
        let _ = queue().lock().unwrap().send(Box::new(|| ()));
    }
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let mut pending = 0;
    PENDING.for_each(|_, p| pending += usize::from(!*p.done.lock().unwrap()));
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}
