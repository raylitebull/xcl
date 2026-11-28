//! A budget for the RAM the process holds (`XCL_RAM_MB`).
//!
//! A console title's GPU memory is its only copy of what is in it. Here the title's pages are
//! RAM, and what the GPU reads of them is copied to video memory: a retail title's world held
//! 5 GB of RAM in pages whose copy was on the card (3.1 GB of streamed physical pages, 1.6 GB
//! of GPU heaps, the coherent window), of 13.8 GB in all, and most of those pages are not
//! touched again once they are uploaded. Windows only takes such pages out of a process when
//! RAM is short for the whole system – on a 32 GB PC with 15 GB in use by other programs that
//! was at under 4 GB free, where a guard had ended the run.
//!
//! With a budget the process's working set has a hard maximum: Windows takes the pages used
//! least out of it as it grows past that. They stay in RAM (the standby and modified lists) while
//! nothing else needs it, and a page touched again comes back with a soft fault; what the
//! system then needs is written to the page file. The process's commit is as before.
//!
//! `XCL_RAM_MB=<MB>`: that maximum. `XCL_RAM_MB=auto`: the maximum follows the RAM the system
//! has available, a second at a time, so that `XCL_RAM_FREE_MB` (6144 unless given) stay
//! available. Unset: no budget.

use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Memory::SetProcessWorkingSetSizeEx;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const QUOTA_LIMITS_HARDWS_MIN_DISABLE: u32 = 0x2;
const QUOTA_LIMITS_HARDWS_MAX_ENABLE: u32 = 0x4;
const QUOTA_LIMITS_HARDWS_MAX_DISABLE: u32 = 0x8;

const MB: u64 = 1 << 20;
/// no budget under this: the title's own threads' pages alone are more
const LEAST: u64 = 4096 * MB;
/// the most an automatic budget moves in one step (a second): pages taken out are available
/// only once they are written to the page file, and the next step would take them out again
const STEP: u64 = 512 * MB;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Setting {
    Fixed(u64),
    /// keep this many bytes available
    Auto(u64),
}

fn setting_of(ram: Option<&str>, free: Option<&str>) -> Option<Setting> {
    let ram = ram?.trim();
    if ram.eq_ignore_ascii_case("auto") {
        let free = free.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(6144);
        return Some(Setting::Auto(free * MB));
    }
    ram.parse::<u64>().ok().filter(|&mb| mb > 0).map(|mb| Setting::Fixed((mb * MB).max(LEAST)))
}

/// The next maximum of an automatic budget: the working set `ws` may grow by what is available
/// beyond `keep`, and has to shrink by what is missing; `before` (None: no maximum yet) moves
/// by `STEP` at most. None: no maximum is needed (more than `STEP` to spare).
fn next_max(before: Option<u64>, ws: u64, available: u64, keep: u64) -> Option<u64> {
    let want = (ws + available).saturating_sub(keep).max(LEAST);
    match before {
        None if available > keep + STEP => None,
        None => Some(want.max(ws.saturating_sub(STEP))),
        Some(b) => Some(want.clamp(b.saturating_sub(STEP), b + STEP).max(LEAST)),
    }
}

fn working_set() -> u64 {
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        c.WorkingSetSize as u64
    }
}

fn available() -> u64 {
    unsafe {
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of_val(&m) as u32;
        GlobalMemoryStatusEx(&mut m);
        m.ullAvailPhys
    }
}

/// The hard maximum of the working set is `max` (None: none). False if Windows refuses.
fn set_max(max: Option<u64>) -> bool {
    let (bytes, flags) = match max {
        Some(m) => (m as usize, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE),
        None => (usize::MAX >> 1, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE),
    };
    // (the minimum is not a hard one: a number below the maximum)
    unsafe { SetProcessWorkingSetSizeEx(GetCurrentProcess(), 64 << 20, bytes.max(128 << 20), flags) != 0 }
}

/// Start the budget `XCL_RAM_MB` asks for (nothing without it).
pub fn start() {
    let Some(setting) = setting_of(std::env::var("XCL_RAM_MB").ok().as_deref(), std::env::var("XCL_RAM_FREE_MB").ok().as_deref()) else { return };
    match setting {
        Setting::Fixed(max) => {
            let ok = set_max(Some(max));
            crate::xlog_info!("RAM budget: XCL_RAM_MB – the process's working set is kept under {} MB{}", max / MB, if ok { "" } else { " (Windows refused)" });
        }
        Setting::Auto(keep) => {
            crate::xlog_info!("RAM budget: XCL_RAM_MB=auto – the process's working set is kept so that {} MB of RAM stay available", keep / MB);
            let _ = std::thread::Builder::new().name("xcl-ram-budget".into()).spawn(move || {
                let mut max: Option<u64> = None;
                let mut ticks = 0u32;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let (ws, avail) = (working_set(), available());
                    let next = next_max(max, ws, avail, keep);
                    if next != max {
                        if !set_max(next) {
                            crate::xlog_warn!("RAM budget: Windows refused a working set maximum of {:?} MB", next.map(|m| m / MB));
                            return;
                        }
                        if max.is_none() || next.is_none() {
                            crate::xlog_info!(
                                "RAM budget: {} MB of RAM available with a working set of {} MB: {}",
                                avail / MB,
                                ws / MB,
                                next.map_or("no maximum".to_string(), |m| format!("a maximum of {} MB from here on", m / MB))
                            );
                        }
                        max = next;
                    }
                    ticks += 1;
                    if ticks % 20 == 0 {
                        if let Some(m) = max {
                            crate::xlog_info!("RAM budget: working set {} MB under a maximum of {} MB, {} MB of RAM available", ws / MB, m / MB, avail / MB);
                        }
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_setting_is_a_number_of_megabytes_or_auto() {
        assert_eq!(setting_of(None, None), None);
        assert_eq!(setting_of(Some("0"), None), None);
        assert_eq!(setting_of(Some("lots"), None), None);
        assert_eq!(setting_of(Some(" 10000 "), None), Some(Setting::Fixed(10000 * MB)));
        assert_eq!(setting_of(Some("100"), None), Some(Setting::Fixed(LEAST)), "not under the least");
        assert_eq!(setting_of(Some("auto"), None), Some(Setting::Auto(6144 * MB)));
        assert_eq!(setting_of(Some("AUTO"), Some("8000")), Some(Setting::Auto(8000 * MB)));
    }

    #[test]
    fn an_automatic_maximum_follows_the_ram_available() {
        let keep = 6144 * MB;
        // plenty available: no maximum
        assert_eq!(next_max(None, 5000 * MB, 12000 * MB, keep), None);
        // what is available comes near what is kept: the working set may grow by the difference
        assert_eq!(next_max(None, 9000 * MB, 6500 * MB, keep), Some(9356 * MB));
        // less available than kept: the working set has to shrink by what is missing, a step at most
        assert_eq!(next_max(Some(12000 * MB), 12000 * MB, 5944 * MB, keep), Some(11800 * MB));
        assert_eq!(next_max(Some(12000 * MB), 12000 * MB, 4000 * MB, keep), Some(12000 * MB - STEP));
        assert_eq!(next_max(None, 12000 * MB, 4000 * MB, keep), Some(12000 * MB - STEP));
        // RAM comes back: the maximum goes up again, a step at a time
        assert_eq!(next_max(Some(10000 * MB), 10000 * MB, 9000 * MB, keep), Some(10000 * MB + STEP));
        // never under the least
        assert_eq!(next_max(Some(LEAST), 4096 * MB, 1000 * MB, keep), Some(LEAST));
    }

    /// Windows takes a hard maximum for this process's working set, and lifts it again.
    #[test]
    fn windows_takes_a_working_set_maximum_and_lifts_it() {
        assert!(set_max(Some(working_set() + 4096 * MB)));
        assert!(set_max(None));
    }
}
