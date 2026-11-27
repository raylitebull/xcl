//! A few worker threads that share one submit's bulk memory work: the scan of written pages and
//! the copies into the staging ring (`sparse::before_execute`). The submitting thread takes part
//! and waits for the workers, so nothing of the submit is left running when `run` returns.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// What one `run` shares: the function and how many calls it makes (one per item).
struct Job {
    /// the caller's closure: valid while `run` waits for every worker that took the job
    f: *const (dyn Fn(usize) + Sync),
    items: usize,
    /// when it was handed out
    start: Instant,
}

/// How a `run`'s items were shared (`XCL_LOOK_STATS=2` reports it for the scan): a caller that
/// did most of them itself, or workers that came late, make the pool worth nothing.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Shares {
    pub by_caller: usize,
    pub by_workers: usize,
    /// how long after the job was handed out the first worker took an item (None: none did)
    pub first: Option<Duration>,
}

// SAFETY: the pointer is only used while `run` (which holds the closure) waits.
unsafe impl Send for Job {}

struct State {
    gen: u64,
    job: Option<Job>,
    /// workers that took the job and have not finished it
    running: usize,
}

struct Shared {
    state: Mutex<State>,
    start: Condvar,
    done: Condvar,
    next: AtomicUsize,
    /// the job's items the workers did, and the ns until the first of them took one (`Shares`)
    by_workers: AtomicUsize,
    first_ns: AtomicU64,
}

struct Pool {
    shared: Arc<Shared>,
}

fn worker(s: Arc<Shared>) {
    let mut seen = 0u64;
    loop {
        let (f, items, start) = {
            let mut st = s.state.lock().unwrap_or_else(|e| e.into_inner());
            while st.gen == seen || st.job.is_none() {
                st = s.start.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            seen = st.gen;
            st.running += 1;
            let j = st.job.as_ref().expect("job");
            (j.f, j.items, j.start)
        };
        let mut did = 0usize;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loop {
            let i = s.next.fetch_add(1, Ordering::Relaxed);
            if i >= items {
                break;
            }
            if did == 0 {
                s.first_ns.fetch_min(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
            did += 1;
            // SAFETY: `run` waits for `running` to reach 0 before its closure goes away
            unsafe { (*f)(i) };
        }));
        // (before `running` goes down: `run` reads it once that is 0)
        s.by_workers.fetch_add(did, Ordering::Relaxed);
        let mut st = s.state.lock().unwrap_or_else(|e| e.into_inner());
        st.running -= 1;
        if st.running == 0 {
            s.done.notify_all();
        }
    }
}

/// `XCL_COPY_THREADS` workers (4 unless given, 0: none), started with the first use.
fn pool() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let n: usize = std::env::var("XCL_COPY_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4).min(15);
        if n == 0 {
            return None;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(State { gen: 0, job: None, running: 0 }),
            start: Condvar::new(),
            done: Condvar::new(),
            next: AtomicUsize::new(0),
            by_workers: AtomicUsize::new(0),
            first_ns: AtomicU64::new(u64::MAX),
        });
        let mut made = 0;
        for i in 0..n {
            let s = Arc::clone(&shared);
            if std::thread::Builder::new().name(format!("xcl-copy-{i}")).spawn(move || worker(s)).is_ok() {
                made += 1;
            }
        }
        (made > 0).then_some(Pool { shared })
    })
    .as_ref()
}

/// `f(0) … f(items - 1)`, on the calling thread and the workers; returns when all have returned,
/// with who did them.
pub(super) fn run(items: usize, f: &(dyn Fn(usize) + Sync)) -> Shares {
    let Some(p) = (items > 1).then(pool).flatten() else {
        (0..items).for_each(f);
        return Shares { by_caller: items, ..Shares::default() };
    };
    // one job at a time
    static ONE: Mutex<()> = Mutex::new(());
    let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the pointer outlives the workers' use of it (`run` waits for them below)
    let erased: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
    {
        let mut st = p.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        p.shared.next.store(0, Ordering::Relaxed);
        p.shared.by_workers.store(0, Ordering::Relaxed);
        p.shared.first_ns.store(u64::MAX, Ordering::Relaxed);
        st.gen += 1;
        st.job = Some(Job { f: erased, items, start: Instant::now() });
    }
    p.shared.start.notify_all();
    let mut by_caller = 0;
    loop {
        let i = p.shared.next.fetch_add(1, Ordering::Relaxed);
        if i >= items {
            break;
        }
        f(i);
        by_caller += 1;
    }
    let mut st = p.shared.state.lock().unwrap_or_else(|e| e.into_inner());
    while st.running > 0 {
        st = p.shared.done.wait(st).unwrap_or_else(|e| e.into_inner());
    }
    st.job = None;
    let first = p.shared.first_ns.load(Ordering::Relaxed);
    Shares { by_caller, by_workers: p.shared.by_workers.load(Ordering::Relaxed), first: (first != u64::MAX).then(|| Duration::from_nanos(first)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_item_runs_once() {
        for items in [0usize, 1, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..items).map(|_| AtomicUsize::new(0)).collect();
            run(items, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "{items} items");
        }
        // many runs one after the other, with the same workers
        for _ in 0..200 {
            let n = AtomicUsize::new(0);
            run(20, &|_| {
                n.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(n.load(Ordering::Relaxed), 20);
        }
    }

    #[test]
    fn the_caller_and_the_workers_did_every_item() {
        for items in [0usize, 1, 2, 7, 64, 1000] {
            for _ in 0..20 {
                let s = run(items, &|_| std::hint::black_box(()));
                assert_eq!(s.by_caller + s.by_workers, items, "{s:?}");
                // (a worker's first item is when it is counted as begun)
                assert_eq!(s.first.is_some(), s.by_workers > 0, "{s:?}");
            }
        }
        // items that take a while: the workers have time to take some
        let s = run(64, &|_| std::thread::sleep(std::time::Duration::from_micros(200)));
        assert_eq!(s.by_caller + s.by_workers, 64, "{s:?}");
    }
}
