//! Diagnostic: where the time of a title's submit (ExecuteCommandLists) goes, as averages over the
//! last 1024 of them (logged; a few relaxed adds a submit).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Cumulative microseconds by stage, and the submits counted.
static SUM: [AtomicU64; STAGES] = [const { AtomicU64::new(0) }; STAGES];
const LOCK: usize = 0;
const REFRESH: usize = 1;
const DRIVER: usize = 2;
const AFTER: usize = 3;
const TOTAL: usize = 4;
const N: usize = 5;
const RECHECK: usize = 6;
const RANGES: usize = 7;
const TEXTURES: usize = 8;
/// what lies between the stages above: the whole call of the refresh (its lock wait and the
/// refresh with what it does around them), the lists made ready for the PC, the waits for a list's
/// previous submit, the whole of the lists' submission (the driver and after stages of each piece
/// with the work between pieces), and the end after it
const CALL: usize = 9;
const LISTS: usize = 10;
const PREVIOUS: usize = 11;
const EXEC: usize = 12;
const END: usize = 13;
/// between the pieces of a submit split at a wait (`list_x::execute_with_waits_on`): the waits,
/// the sparse windows brought up to date again, and how many splits (a count)
const SPLIT_WAITS: usize = 14;
const SPLIT_SPARSE: usize = 15;
const SPLITS: usize = 16;
const STAGES: usize = 17;

pub(super) fn add(stage: usize, d: Duration) {
    SUM[stage].fetch_add(d.as_micros() as u64, Ordering::Relaxed);
}

pub(super) fn call(d: Duration) {
    add(CALL, d);
}

pub(super) fn lists(d: Duration) {
    add(LISTS, d);
}

pub(super) fn previous(d: Duration) {
    add(PREVIOUS, d);
}

pub(super) fn exec(d: Duration) {
    add(EXEC, d);
}

pub(super) fn end(d: Duration) {
    add(END, d);
}

/// A submit was split at a wait: the wait's time, and that of the sparse windows' update after it.
pub(super) fn split(waits: Duration, sparse: Duration) {
    add(SPLIT_WAITS, waits);
    add(SPLIT_SPARSE, sparse);
    SUM[SPLITS].fetch_add(1, Ordering::Relaxed);
}

/// The part of the average line for what lies outside the stages it names, from the averages a
/// submit (`ms`, ms; `SPLITS` as a count a submit): the refresh's call around its lock wait and
/// itself, the lists made ready, the waits for previous submits, the splits (the submission but its
/// driver and after stages and those waits), and the end.
fn outside_part(ms: &[f64; STAGES]) -> String {
    let entry = (ms[CALL] - ms[LOCK] - ms[REFRESH]).max(0.0);
    let splits = (ms[EXEC] - ms[DRIVER] - ms[AFTER] - ms[PREVIOUS]).max(0.0);
    format!(
        "outside them: entry {:.2}, lists {:.2}, previous submits {:.2}, splits {:.2} (waits {:.2}, sparse again {:.2}; {:.2} a submit), end {:.2}",
        entry, ms[LISTS], ms[PREVIOUS], splits, ms[SPLIT_WAITS], ms[SPLIT_SPARSE], ms[SPLITS], ms[END]
    )
}

/// the wait for the refresh lock, and the refresh itself
pub(super) fn lock_wait(d: Duration) {
    add(LOCK, d);
}

pub(super) fn refresh(d: Duration) {
    add(REFRESH, d);
}

/// the PC ExecuteCommandLists of a chunk of the title's lists, and everything XCL does after it
pub(super) fn driver(d: Duration) {
    add(DRIVER, d);
}

pub(super) fn after(d: Duration) {
    add(AFTER, d);
}

/// parts of the refresh: re-checking the sources in GPU views, the watched ranges, the textures'
/// uploads (beside the parts `sparse` counts)
pub(super) fn recheck(d: Duration) {
    add(RECHECK, d);
}

pub(super) fn ranges(d: Duration) {
    add(RANGES, d);
}

pub(super) fn textures(d: Duration) {
    add(TEXTURES, d);
}

/// The texture uploads: [looked at, unchanged, uploaded, bytes compared, bytes uploaded, µs
/// comparing, batches, µs uploading].
static TEX: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Textures with written pages were compared with what those held.
pub(super) fn textures_compared(textures: usize, unchanged: usize, bytes: usize, took: Duration) {
    TEX[0].fetch_add(textures as u64, Ordering::Relaxed);
    TEX[1].fetch_add(unchanged as u64, Ordering::Relaxed);
    TEX[3].fetch_add(bytes as u64, Ordering::Relaxed);
    TEX[5].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
}

/// A batch of textures, or of parts of them, went up.
pub(super) fn textures_uploaded(textures: usize, bytes: usize, took: Duration) {
    TEX[2].fetch_add(textures as u64, Ordering::Relaxed);
    TEX[4].fetch_add(bytes as u64, Ordering::Relaxed);
    TEX[6].fetch_add(1, Ordering::Relaxed);
    TEX[7].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
}

/// A whole submit (from the hook's entry) is done.
pub(super) fn done(t0: Instant) {
    add(TOTAL, t0.elapsed());
    // `XCL_LOOK_STATS=2`: the title's submits for `api_cost` (and the thread that makes them),
    // not XCL's own ones inside a submit or a present: those are in that one's time
    if super::look_stats::deep() && !super::title_textures::is_own_work() && !super::present::presenting() {
        crate::hle::api_cost::note_submitter();
        crate::hle::api_cost::add(crate::hle::api_cost::Call::Submit, t0.elapsed());
    }
    let n = SUM[N].fetch_add(1, Ordering::Relaxed) + 1;
    if n % 1024 != 0 {
        return;
    }
    struct Last {
        sum: [u64; STAGES],
        parts: [u64; 4],
        tex: [u64; 8],
        at: Instant,
    }
    static LAST: Mutex<Option<Last>> = Mutex::new(None);
    let mut g = LAST.lock().unwrap();
    let now: [u64; STAGES] = std::array::from_fn(|i| SUM[i].load(Ordering::Relaxed));
    // [look, STATE lock wait, chunk scan, upload] µs inside the refreshes
    let parts = super::sparse::timing_totals();
    let tex: [u64; 8] = std::array::from_fn(|i| TEX[i].load(Ordering::Relaxed));
    if let Some(l) = g.as_ref() {
        let x = |i: usize| tex[i].saturating_sub(l.tex[i]);
        crate::xlog_info!(
            "submit cost, textures in these 1024: {} with written pages ({} unchanged), {:.1} MB compared in {:.1} ms; {} uploaded, {:.1} MB in {} batches, {:.1} ms",
            x(0),
            x(1),
            x(3) as f64 / 1e6,
            x(5) as f64 / 1e3,
            x(2),
            x(4) as f64 / 1e6,
            x(6),
            x(7) as f64 / 1e3
        );
        let d = |i: usize| now[i].saturating_sub(l.sum[i]) as f64 / 1024.0 / 1000.0;
        let p = |i: usize| parts[i].saturating_sub(l.parts[i]) as f64 / 1024.0 / 1000.0;
        let refresh = d(REFRESH);
        let mut ms: [f64; STAGES] = std::array::from_fn(d);
        ms[SPLITS] = now[SPLITS].saturating_sub(l.sum[SPLITS]) as f64 / 1024.0;
        crate::xlog_info!(
            "submit cost, average of 1024 ExecuteCommandLists: {:.2} ms in all = {:.2} refresh lock wait + {:.2} refresh (look {:.2}, ranges {:.2}, recheck {:.2}, sparse lock {:.2}, scan {:.2}, upload {:.2}, textures {:.2}, rest {:.2}) + {:.2} driver + {:.2} after ({:.0} submits/s); {}",
            d(TOTAL),
            d(LOCK),
            refresh,
            p(0),
            d(RANGES),
            d(RECHECK),
            p(1),
            p(2),
            p(3),
            d(TEXTURES),
            refresh - p(0) - p(1) - p(2) - p(3) - d(RANGES) - d(RECHECK) - d(TEXTURES),
            d(DRIVER),
            d(AFTER),
            1024.0 / l.at.elapsed().as_secs_f64().max(1e-3),
            outside_part(&ms)
        );
    }
    *g = Some(Last { sum: now, parts, tex, at: Instant::now() });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The part for what lies outside the named stages: the refresh's call less its lock wait and
    /// itself, the submission less its driver, after and previous-submit parts; never below 0.
    #[test]
    fn the_outside_part_is_what_the_named_stages_leave() {
        let mut ms = [0.0f64; STAGES];
        ms[LOCK] = 0.5;
        ms[REFRESH] = 5.0;
        ms[CALL] = 5.75;
        ms[LISTS] = 0.1;
        ms[DRIVER] = 0.25;
        ms[AFTER] = 1.0;
        ms[PREVIOUS] = 0.5;
        ms[EXEC] = 3.0;
        ms[SPLIT_WAITS] = 0.25;
        ms[SPLIT_SPARSE] = 1.0;
        ms[SPLITS] = 0.5;
        ms[END] = 0.2;
        assert_eq!(outside_part(&ms), "outside them: entry 0.25, lists 0.10, previous submits 0.50, splits 1.25 (waits 0.25, sparse again 1.00; 0.50 a submit), end 0.20");
        ms[CALL] = 1.0;
        ms[EXEC] = 0.0;
        assert!(outside_part(&ms).starts_with("outside them: entry 0.00, lists 0.10, previous submits 0.50, splits 0.00"));
    }
}
