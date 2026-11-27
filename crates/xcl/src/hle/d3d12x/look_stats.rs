//! Diagnostic (`XCL_LOOK_STATS=1`): what the looks for CPU writes find, as averages over 256 of them
//! (logged): how many of a look's pages the looks before it had found too, in how many runs they
//! lie, how many of the pages scanned had changed, and how long before the submit that looked the
//! youngest of its lists had been closed.
//!
//! `XCL_LOOK_STATS=2` adds the splits (`deep`): where the time of a look, of the scan after it and
//! of the submitting thread goes, finer than `submit_cost`'s line, to choose between the fixes of
//! a slow submit path.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// `XCL_LOOK_STATS`: 1 the lines of the looks, 2 those and the splits; 0 when unset
fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("XCL_LOOK_STATS").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

pub(super) fn on() -> bool {
    level() >= 1
}

/// `XCL_LOOK_STATS=2`: the splits are measured too
pub(super) fn deep() -> bool {
    level() >= 2
}

/// the upper ends (µs) of the steps `Stats::ago` counts by; the last step has none
const STEPS: [u64; 6] = [20, 100, 500, 2_000, 10_000, 50_000];

#[derive(Default)]
struct Stats {
    looks: u64,
    us: u64,
    pages: u64,
    /// of them found between the submits (`XCL_LOOK_AHEAD`)
    ahead: u64,
    runs: u64,
    /// pages also in the look before, and in all three before
    once: u64,
    thrice: u64,
    /// pages scanned (the ones of video memory's windows), and those with a changed chunk
    scanned: u64,
    changed: u64,
    ago: [u64; STEPS.len() + 1],
    /// the pages of the three looks before, the last one first
    before: [Vec<usize>; 3],
    /// the looks' write-watch calls (`deep`, `add_call`)
    split: [u64; 8],
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

thread_local! {
    /// the write-watch calls of this thread's look so far (`add_call`): a look's calls are all
    /// made on its submitting thread, and `look` takes them
    static CALLS: Cell<[u64; 8]> = const { Cell::new([0; 8]) };
}

/// One write-watch call into `acc`: [ns, bytes covered, pages, calls] of the calls that returned
/// no page, then the same of those that returned some (ns: a call over little memory takes a few
/// µs). A look over memory nobody wrote costs a walk of its page tables, one with pages a
/// page-table entry changed for each: which of the two a look's time is decides between looking
/// at less memory and taking fewer pages.
fn add_call(acc: &mut [u64; 8], ns: u64, bytes: u64, pages: u64) {
    let k = if pages == 0 { 0 } else { 4 };
    acc[k] += ns;
    acc[k + 1] += bytes;
    acc[k + 2] += pages;
    acc[k + 3] += 1;
}

/// Sums of a `deep` line, added from any thread with relaxed adds (no lock: some are added with
/// the sparse state's held); `take` gives them and starts again (an add made while they are taken
/// may go to the next line's).
pub(super) struct Sums<const N: usize>([AtomicU64; N]);

impl<const N: usize> Sums<N> {
    pub(super) const fn new() -> Self {
        Sums([const { AtomicU64::new(0) }; N])
    }

    /// (the sum with it)
    pub(super) fn add(&self, k: usize, v: u64) -> u64 {
        self.0[k].fetch_add(v, Ordering::Relaxed) + v
    }

    pub(super) fn max(&self, k: usize, v: u64) {
        self.0[k].fetch_max(v, Ordering::Relaxed);
    }

    pub(super) fn take(&self) -> [u64; N] {
        std::array::from_fn(|k| self.0[k].swap(0, Ordering::Relaxed))
    }
}

/// How stale a look can be (`walk_stale`, `walk_pass`): [looks, looks that began with a block
/// the walk found that no look had taken, walk passes, ns walking, the longest pass's ns]. A look
/// takes the blocks written lately only; a write to another block waits for the walk to find it.
static STALE: Sums<5> = Sums::new();

/// A look began (`deep` only); `stale`: the walk had made a block hot since the look before began.
pub(super) fn walk_stale(stale: bool) {
    STALE.add(0, 1);
    STALE.add(1, stale as u64);
}

/// A pass of the walk over all watched memory took `took` (`deep` only).
pub(super) fn walk_pass(took: Duration) {
    STALE.add(2, 1);
    STALE.add(3, took.as_nanos() as u64);
    STALE.max(4, took.as_nanos() as u64);
}

/// The pages XCL wrote into the title's memory itself (`own_writes`): [looks, pages written since
/// the look before (copies back landed, immediates stored), those in the look, those scanned,
/// those the scan found unchanged]. The write watch reports them like the title's writes, and a
/// look and a scan spent on them is spent on nothing new.
static OWN: Sums<5> = Sums::new();

/// What a look's submit found of XCL's own writes since the look before (`deep` only).
pub(super) fn own_writes(written: usize, in_look: usize, scanned: usize, unchanged: usize) {
    OWN.add(0, 1);
    OWN.add(1, written as u64);
    OWN.add(2, in_look as u64);
    OWN.add(3, scanned as u64);
    OWN.add(4, unchanged as u64);
}

/// The scans (`scan_split`): [submits, ns gathering, ns scanning, jobs shared, their items, those
/// the caller did, ns until a job's first worker began, jobs a worker began]
static SCAN: Sums<8> = Sums::new();

/// A submit's scan (`deep` only): `gather` building its pages' slots, `scan` the scan of them,
/// and how that was shared with the workers (None: too few pages to share).
pub(super) fn scan_split(gather: Duration, scan: Duration, shares: Option<super::worker_pool::Shares>) {
    SCAN.add(0, 1);
    SCAN.add(1, gather.as_nanos() as u64);
    SCAN.add(2, scan.as_nanos() as u64);
    if let Some(s) = shares {
        SCAN.add(3, 1);
        // (the items done: all of the job's)
        SCAN.add(4, (s.by_caller + s.by_workers) as u64);
        SCAN.add(5, s.by_caller as u64);
        if let Some(d) = s.first {
            SCAN.add(6, d.as_nanos() as u64);
            SCAN.add(7, 1);
        }
    }
}

/// A write-watch call of a look (`deep` only) covered `bytes`, took `took` and returned `pages`.
pub(super) fn watch_call(took: Duration, bytes: usize, pages: usize) {
    CALLS.with(|c| {
        let mut acc = c.get();
        add_call(&mut acc, took.as_nanos() as u64, bytes as u64, pages as u64);
        c.set(acc);
    });
}

/// How many of `a`'s pages every one of `others` has too (all sorted, each page once).
fn in_all(a: &[usize], others: &[Vec<usize>]) -> u64 {
    let mut at = vec![0usize; others.len()];
    a.iter()
        .filter(|&&p| {
            others.iter().zip(at.iter_mut()).all(|(o, k)| {
                while *k < o.len() && o[*k] < p {
                    *k += 1;
                }
                o.get(*k) == Some(&p)
            })
        })
        .count() as u64
}

/// The runs of neighbouring pages of `pages` (sorted, 4 KiB each).
fn runs_of(pages: &[usize]) -> u64 {
    pages.iter().zip(std::iter::once(&usize::MAX).chain(pages)).filter(|(p, q)| **p != q.wrapping_add(0x1000)).count() as u64
}

/// A look found `pages` (sorted), `ahead` of them before its submit; it took `took`. `closed`: how
/// long before its submit the youngest of the submit's lists was closed (None: not known).
pub(super) fn look(pages: &[usize], ahead: usize, took: Duration, closed: Option<u64>) {
    if !on() {
        return;
    }
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(Stats::default);
    s.looks += 1;
    s.us += took.as_micros() as u64;
    s.pages += pages.len() as u64;
    s.ahead += ahead as u64;
    s.runs += runs_of(pages);
    s.once += in_all(pages, &s.before[..1]);
    s.thrice += in_all(pages, &s.before);
    s.before.rotate_right(1);
    s.before[0].clear();
    s.before[0].extend_from_slice(pages);
    if let Some(us) = closed {
        s.ago[STEPS.partition_point(|&e| e <= us)] += 1;
    }
    if deep() {
        let calls = CALLS.with(|c| c.replace([0; 8]));
        s.split.iter_mut().zip(calls).for_each(|(a, b)| *a += b);
    }
    if s.looks % 256 == 0 && deep() {
        let x = &s.split;
        crate::xlog_info!(
            "D3D12.X: look split (256 looks): {:.2} ms a look in {:.1} calls over {} MB that found nothing + {:.2} ms in {:.1} calls over {} MB with {} pages",
            x[0] as f64 / 256e6,
            x[3] as f64 / 256.0,
            (x[1] / 256) >> 20,
            x[4] as f64 / 256e6,
            x[7] as f64 / 256.0,
            (x[5] / 256) >> 20,
            x[6] / 256
        );
        let c = SCAN.take();
        let per = |v: u64, of: u64| v as f64 / of.max(1) as f64;
        crate::xlog_info!(
            "D3D12.X: scan split (256 looks): gather {:.2} ms, scan {:.2} ms a submit; of {:.0} items a job the caller did {:.0} %, the first worker began after {:.0} µs ({} jobs in {} submits, {} with a worker)",
            per(c[1], c[0]) / 1e6,
            per(c[2], c[0]) / 1e6,
            per(c[4], c[3]),
            per(c[5] * 100, c[4]),
            per(c[6], c[7]) / 1e3,
            c[3],
            c[0],
            c[7]
        );
        let w = STALE.take();
        crate::xlog_info!(
            "D3D12.X: walk staleness (256 looks): {} of {} looks began with a block the walk found that no look had taken yet; a walk pass took {:.1} ms ({:.1} ms the longest, {} passes)",
            w[1],
            w[0],
            per(w[3], w[2]) / 1e6,
            w[4] as f64 / 1e6,
            w[2]
        );
        let o = OWN.take();
        crate::xlog_info!(
            "D3D12.X: XCL's own writes (256 looks): {:.1} pages a look written by XCL since the look before (copies back landed, immediates stored), {:.1} of them in the look, {:.1} scanned, {:.0} % of those found unchanged",
            per(o[1], o[0]),
            per(o[2], o[0]),
            per(o[3], o[0]),
            per(o[4] * 100, o[3])
        );
    }
    if s.looks % 256 == 0 {
        let a = |v: u64| v / 256;
        let share = |v: u64, of: u64| if of == 0 { 0 } else { v * 100 / of };
        crate::xlog_info!(
            "D3D12.X: the last 256 looks for CPU writes: {:.2} ms and {} pages a look ({} found ahead) in {} runs ({} % of the pages also in the look before, {} % in all three before); {} scanned a look, {} % of them changed; youngest list closed before the submit: {} <20 µs, {} <100 µs, {} <500 µs, {} <2 ms, {} <10 ms, {} <50 ms, {} longer",
            s.us as f64 / 256e3,
            a(s.pages),
            a(s.ahead),
            a(s.runs),
            share(s.once, s.pages),
            share(s.thrice, s.pages),
            a(s.scanned),
            share(s.changed, s.scanned),
            s.ago[0],
            s.ago[1],
            s.ago[2],
            s.ago[3],
            s.ago[4],
            s.ago[5],
            s.ago[6]
        );
        *s = Stats { looks: s.looks, before: std::mem::take(&mut s.before), ..Stats::default() };
    }
}

/// A scan went over `pages` pages and found the changed `runs` (address, length; sorted).
pub(super) fn scan(pages: usize, runs: &[(usize, usize)]) {
    if !on() {
        return;
    }
    let mut g = STATS.lock().unwrap_or_else(|e| e.into_inner());
    let s = g.get_or_insert_with(Stats::default);
    s.scanned += pages as u64;
    let mut next = 0usize;
    for &(a, n) in runs {
        let (first, end) = ((a & !0xFFF).max(next), (a + n + 0xFFF) & !0xFFF);
        if end > first {
            s.changed += ((end - first) >> 12) as u64;
            next = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_found_again_and_their_runs() {
        let p = |v: &[usize]| v.iter().map(|k| k * 0x1000).collect::<Vec<_>>();
        let a = p(&[1, 2, 3, 7, 9, 10]);
        assert_eq!(runs_of(&a), 3);
        assert_eq!(runs_of(&[]), 0);
        assert_eq!(runs_of(&p(&[1])), 1);
        assert_eq!(in_all(&a, &[p(&[2, 3, 4, 10])]), 3);
        assert_eq!(in_all(&a, &[p(&[2, 3, 4, 10]), p(&[3, 7, 10]), p(&[1, 3])]), 1);
        assert_eq!(in_all(&a, &[p(&[])]), 0);
    }

    #[test]
    fn a_looks_calls_add_up_to_what_was_put_in() {
        // (ns, bytes, pages) of the calls of one look
        let calls = [(4_000u64, 64u64 << 20, 0u64), (9_000, 2 << 20, 37), (1_500, 4 << 20, 0), (700, 1 << 20, 1)];
        let mut acc = [0u64; 8];
        for &(ns, bytes, pages) in &calls {
            add_call(&mut acc, ns, bytes, pages);
        }
        let sum = |f: fn(&(u64, u64, u64)) -> u64| calls.iter().map(f).sum::<u64>();
        assert_eq!(acc[0] + acc[4], sum(|c| c.0));
        assert_eq!(acc[1] + acc[5], sum(|c| c.1));
        assert_eq!(acc[2] + acc[6], sum(|c| c.2));
        assert_eq!(acc[3] + acc[7], calls.len() as u64);
        // the calls that found nothing on one side, the others on the other
        assert_eq!(acc[..4], [5_500, 68 << 20, 0, 2]);
        assert_eq!(acc[4..], [9_700, 3 << 20, 38, 2]);
        // a thread's calls go to its own look
        watch_call(Duration::from_micros(3), 8192, 2);
        std::thread::spawn(|| watch_call(Duration::from_micros(5), 4096, 0)).join().unwrap();
        assert_eq!(CALLS.with(|c| c.replace([0; 8])), [0, 0, 0, 0, 3_000, 8192, 2, 1]);
    }

    #[test]
    fn sums_from_threads_are_taken_once() {
        static S: Sums<3> = Sums::new();
        std::thread::scope(|t| {
            for k in 0..4u64 {
                t.spawn(move || {
                    for _ in 0..1000 {
                        assert!(S.add(0, 1) >= 1);
                        S.add(2, k);
                    }
                });
            }
        });
        assert_eq!(S.take(), [4000, 0, 6000]);
        assert_eq!(S.take(), [0, 0, 0]);
    }

    #[test]
    fn staleness_and_own_writes_are_summed() {
        // (only `deep` runs add to these, and no test is: this one has them alone)
        for stale in [true, false, true] {
            walk_stale(stale);
        }
        walk_pass(Duration::from_millis(3));
        walk_pass(Duration::from_millis(7));
        assert_eq!(STALE.take(), [3, 2, 2, 10_000_000, 7_000_000]);
        own_writes(40, 30, 28, 25);
        own_writes(2, 0, 0, 0);
        assert_eq!(OWN.take(), [2, 42, 30, 28, 25]);
    }
}
