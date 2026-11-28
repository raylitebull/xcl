//! Where the process's memory is (`XCL_MEM_MAP=1`): every 20 s, with the memory lines of the
//! activity log, a walk of the process's own address space adds up the committed and the resident
//! (working-set) bytes by whose they are: the title's in the UMA windows, the physical pages of
//! `XCL_PAGE_ALIAS`, the images, and the rest by the size of its regions, with the largest
//! allocations of the rest by address.
//!
//! The process's counters (`memory: N MB committed by the process`) hold the title's memory, the
//! driver's, Direct3D's and XCL's own in one number. The walk tells them apart by address and by
//! the type of each region (`VirtualQuery`), and finds the pages that are resident with
//! `K32QueryWorkingSetEx`, `CHUNK` pages a call. It only asks Windows: no page is touched.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED};
use windows_sys::Win32::System::ProcessStatus::{K32GetMappedFileNameW, K32GetProcessMemoryInfo, K32QueryWorkingSetEx, PROCESS_MEMORY_COUNTERS_EX, PSAPI_WORKING_SET_EX_INFORMATION};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// `XCL_MEM_MAP=1` turns the report on (default: off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_MEM_MAP").is_ok_and(|v| v.trim() == "1"))
}

/// the pages the working set is asked about
const PAGE: usize = 4 << 10;
/// pages one `K32QueryWorkingSetEx` call asks about
const CHUNK: usize = 4096;

// Whose committed memory is, the first that fits: by its address (the UMA windows, the physical
// pages' spaces and homes), then by the type of its region, the private rest by the region's size.
const SPARSE: usize = 0;
const COHERENT: usize = 1;
const VIEW: usize = 2;
const SPACES: usize = 3;
const HOMES: usize = 4;
const IMAGE: usize = 5;
const MAPPED: usize = 6;
const PRIVATE_LARGE: usize = 7;
const PRIVATE_MID: usize = 8;
const PRIVATE_SMALL: usize = 9;
const CLASSES: usize = 10;
const NAMES: [&str; CLASSES] = [
    "title memory in sparse windows",
    "title memory in the coherent window",
    "GPU-view windows",
    "physical pages: views in the title's spaces (views of the same pages as the homes)",
    "physical pages: XCL's home views",
    "images (exe and DLLs)",
    "other mapped (sections, files)",
    "other private, regions >= 64 MiB",
    "other private, regions 1-64 MiB",
    "other private, regions < 1 MiB",
];

/// the largest images listed
const TOP_IMAGES: usize = 3;
/// the largest other private allocations listed
const TOP_ALLOCS: usize = 12;

#[derive(Clone, Copy, Default)]
struct Bytes {
    committed: usize,
    resident: usize,
    /// of `resident`, pages mapped more than once (`ShareCount` > 1)
    shared: usize,
    /// committed regions
    regions: usize,
}

impl Bytes {
    fn resident_page(&mut self, shared: bool) {
        self.resident += PAGE;
        if shared {
            self.shared += PAGE;
        }
    }
}

/// One allocation of the walk (its regions' `AllocationBase`): an image, or other private memory.
struct Item {
    base: usize,
    /// the end of its last region seen, committed or reserved
    end: usize,
    bytes: Bytes,
    /// `Protect` of its first committed region; `mixed`: another one has another
    protect: u32,
    mixed: bool,
    alloc_protect: u32,
}

/// Whose a page of the walk is: a class, and an item of the class (`NO_ITEM`: none).
#[derive(Clone, Copy)]
struct Tag {
    class: usize,
    item: usize,
}

const NO_ITEM: usize = usize::MAX;

/// What a walk adds up: whose each committed region is, and its pages found resident.
trait Sink {
    /// The committed part `[at, at + len)` of the region `m`: whose it is.
    fn committed(&mut self, m: &MEMORY_BASIC_INFORMATION, at: usize, len: usize) -> Tag;
    /// A page of `tag`'s is in the working set (`shared`: mapped more than once).
    fn resident(&mut self, tag: Tag, shared: bool);
    /// A region reserved but not committed.
    fn reserved(&mut self, _m: &MEMORY_BASIC_INFORMATION) {}
}

/// The pages of committed regions waiting for one `K32QueryWorkingSetEx` call (one buffer for the
/// whole walk), with whose each one is.
struct Pages {
    info: Vec<PSAPI_WORKING_SET_EX_INFORMATION>,
    tags: Vec<Tag>,
    /// calls Windows refused (their pages are not counted resident)
    failed: usize,
}

impl Pages {
    fn new() -> Pages {
        Pages { info: Vec::with_capacity(CHUNK), tags: Vec::with_capacity(CHUNK), failed: 0 }
    }

    /// The pages of `[at, at + len)`, committed and `tag`'s, asked about `CHUNK` at a time.
    fn add(&mut self, at: usize, len: usize, tag: Tag, sink: &mut impl Sink) {
        for p in (at..at + len).step_by(PAGE) {
            if self.info.len() == CHUNK {
                self.flush(sink);
            }
            self.info.push(PSAPI_WORKING_SET_EX_INFORMATION { VirtualAddress: p as *mut c_void, VirtualAttributes: Default::default() });
            self.tags.push(tag);
        }
    }

    /// One `K32QueryWorkingSetEx` call for the pages waiting: each resident one goes to `sink`.
    fn flush(&mut self, sink: &mut impl Sink) {
        if self.info.is_empty() {
            return;
        }
        let bytes = (self.info.len() * std::mem::size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32;
        if unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), self.info.as_mut_ptr() as *mut c_void, bytes) } != 0 {
            for (i, &tag) in self.info.iter().zip(&self.tags) {
                // bit 0: Valid (in the working set), bits 1..=3: ShareCount
                let flags = unsafe { i.VirtualAttributes.Flags };
                if flags & 1 != 0 {
                    sink.resident(tag, (flags >> 1) & 7 > 1);
                }
            }
        } else {
            self.failed += 1;
        }
        self.info.clear();
        self.tags.clear();
    }
}

/// Walk `[lo, hi)` region by region (`VirtualQuery`): the committed part of each goes to `sink`,
/// and its pages to the working-set queries of `pages`; a region not committed costs its query
/// only. The regions walked.
fn walk(lo: usize, hi: usize, pages: &mut Pages, sink: &mut impl Sink) -> usize {
    let mut walked = 0;
    let mut p = lo;
    while p < hi {
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(p as *const c_void, &mut m, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            // past the user address space
            break;
        }
        walked += 1;
        let base = m.BaseAddress as usize;
        let end = base.saturating_add(m.RegionSize);
        if m.State == MEM_COMMIT {
            let (from, to) = (base.max(lo), end.min(hi));
            if to > from {
                let tag = sink.committed(&m, from, to - from);
                pages.add(from, to - from, tag, sink);
            }
        } else if m.State != MEM_FREE {
            sink.reserved(&m);
        }
        if end <= p {
            break;
        }
        p = end;
    }
    pages.flush(sink);
    walked
}

/// Whether `p` lies in one of `ranges` ((base, bytes), by address, not overlapping).
fn within(ranges: &[(usize, usize)], p: usize) -> bool {
    let i = ranges.partition_point(|r| r.0 <= p);
    i > 0 && p - ranges[i - 1].0 < ranges[i - 1].1
}

/// The item of `m`'s allocation in `items` with `len` committed bytes more: the last one, or a new
/// one (the walk goes by address, and the regions of an allocation are side by side).
fn item_of(items: &mut Vec<Item>, m: &MEMORY_BASIC_INFORMATION, len: usize) -> usize {
    let base = m.AllocationBase as usize;
    if items.last().is_none_or(|i| i.base != base) {
        items.push(Item { base, end: base, bytes: Bytes::default(), protect: m.Protect, mixed: false, alloc_protect: m.AllocationProtect });
    }
    let n = items.len() - 1;
    let i = &mut items[n];
    i.bytes.committed += len;
    i.bytes.regions += 1;
    i.mixed |= i.protect != m.Protect;
    i.end = (m.BaseAddress as usize).saturating_add(m.RegionSize);
    n
}

/// The report's sink: the classes, and the images and other private allocations one by one.
#[derive(Default)]
struct Map {
    uma: Option<&'static crate::gpu::uma::Uma>,
    /// the physical pages' spaces and homes (`alias::ranges`), by address
    spaces: Vec<(usize, usize)>,
    homes: Vec<(usize, usize)>,
    classes: [Bytes; CLASSES],
    /// committed bytes by the type of their region: private, mapped, image
    by_type: [usize; 3],
    /// the images and the other private allocations, by address
    images: Vec<Item>,
    allocs: Vec<Item>,
}

impl Map {
    fn class_of(&self, m: &MEMORY_BASIC_INFORMATION, at: usize, len: usize) -> usize {
        if let Some(u) = self.uma {
            if u.is_sparse(at) {
                return SPARSE;
            }
            if u.is_committed(at) {
                return COHERENT;
            }
            if u.is_view(at) {
                return VIEW;
            }
        }
        if within(&self.spaces, at) {
            SPACES
        } else if within(&self.homes, at) {
            HOMES
        } else if m.Type == MEM_IMAGE {
            IMAGE
        } else if m.Type == MEM_MAPPED {
            MAPPED
        } else if len >= 64 << 20 {
            PRIVATE_LARGE
        } else if len >= 1 << 20 {
            PRIVATE_MID
        } else {
            PRIVATE_SMALL
        }
    }

    /// The items kept one by one of `class` (None: the class is only added up).
    fn items(&mut self, class: usize) -> Option<&mut Vec<Item>> {
        match class {
            IMAGE => Some(&mut self.images),
            PRIVATE_LARGE | PRIVATE_MID | PRIVATE_SMALL => Some(&mut self.allocs),
            _ => None,
        }
    }
}

impl Sink for Map {
    fn committed(&mut self, m: &MEMORY_BASIC_INFORMATION, at: usize, len: usize) -> Tag {
        self.by_type[match m.Type {
            MEM_MAPPED => 1,
            MEM_IMAGE => 2,
            _ => 0,
        }] += len;
        let class = self.class_of(m, at, len);
        self.classes[class].committed += len;
        self.classes[class].regions += 1;
        let item = self.items(class).map_or(NO_ITEM, |items| item_of(items, m, len));
        Tag { class, item }
    }

    fn resident(&mut self, tag: Tag, shared: bool) {
        self.classes[tag.class].resident_page(shared);
        if tag.item != NO_ITEM {
            if let Some(items) = self.items(tag.class) {
                items[tag.item].bytes.resident_page(shared);
            }
        }
    }

    fn reserved(&mut self, m: &MEMORY_BASIC_INFORMATION) {
        // (the reserved rest of an allocation listed: a thread stack, a heap's segment)
        if let Some(i) = self.allocs.last_mut().filter(|i| i.base == m.AllocationBase as usize) {
            i.end = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        }
    }
}

/// The file name of the image at `base` (its mapped file's, without the path).
fn image_name(base: usize) -> String {
    let mut buf = [0u16; 512];
    let n = unsafe { K32GetMappedFileNameW(GetCurrentProcess(), base as *const c_void, buf.as_mut_ptr(), buf.len() as u32) } as usize;
    let path = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
    path.rsplit('\\').next().filter(|s| !s.is_empty()).unwrap_or("?").to_string()
}

/// The lines of the activity log, each starting `mem map:`: the totals and how long the walk
/// took, each class, the largest images, the largest other private allocations. None: off.
pub fn lines() -> Option<Vec<String>> {
    if !enabled() {
        return None;
    }
    let t = std::time::Instant::now();
    let (spaces, homes) = crate::hle::kernelx::alias::ranges();
    let mut map = Map { uma: crate::gpu::uma::get(), spaces, homes, ..Map::default() };
    let mut pages = Pages::new();
    let walked = walk(0, usize::MAX, &mut pages, &mut map);
    let ms = t.elapsed().as_millis();
    let (private_usage, working_set) = unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        c.cb = std::mem::size_of_val(&c) as u32;
        K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
        (c.PrivateUsage, c.WorkingSetSize)
    };
    let mb = |b: usize| b >> 20;
    let [private, mapped, image] = map.by_type;
    let resident: usize = map.classes.iter().map(|c| c.resident).sum();
    let failed = if pages.failed != 0 { format!("; {} working-set queries failed (their pages are not counted resident)", pages.failed) } else { String::new() };
    let mut out = vec![format!(
        "mem map: walked {walked} regions in {ms} ms: committed {} MB private, {} MB mapped, {} MB image ({} MB in all); {} MB resident; the process's counters: {} MB private commit, {} MB working set{failed}",
        mb(private),
        mb(mapped),
        mb(image),
        mb(private + mapped + image),
        mb(resident),
        mb(private_usage),
        mb(working_set)
    )];
    for (name, b) in NAMES.iter().zip(&map.classes) {
        out.push(format!("mem map: {name}: {} MB committed, {} MB resident, {} regions, {} MB of the resident shared", mb(b.committed), mb(b.resident), b.regions, mb(b.shared)));
    }
    map.images.sort_by_key(|i| std::cmp::Reverse(i.bytes.committed));
    for i in map.images.iter().take(TOP_IMAGES) {
        out.push(format!("mem map: image {} at {:#x}: {} MB committed, {} MB resident", image_name(i.base), i.base, mb(i.bytes.committed), mb(i.bytes.resident)));
    }
    map.allocs.sort_by_key(|i| std::cmp::Reverse(i.bytes.committed));
    for (k, i) in map.allocs.iter().take(TOP_ALLOCS).enumerate() {
        out.push(format!(
            "mem map: other private allocation {} at {:#x}: {} MB committed, {} MB resident, {} regions, {} MB reserved, protect {:#x}{} (allocation protect {:#x})",
            k + 1,
            i.base,
            mb(i.bytes.committed),
            mb(i.bytes.resident),
            i.bytes.regions,
            mb(i.end - i.base),
            i.protect,
            if i.mixed { " and others" } else { "" },
            i.alloc_protect
        ));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    /// A private allocation of which the first half is committed, and every other page of that
    /// touched: the walk over it counts the committed half in one region of one allocation, the
    /// touched pages resident and not the others.
    #[test]
    fn a_walk_counts_the_committed_bytes_and_the_touched_pages_resident() {
        // (more pages than one working-set query asks about)
        const SIZE: usize = 20 << 20;
        let p = unsafe { VirtualAlloc(std::ptr::null(), 2 * SIZE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        assert_eq!(unsafe { VirtualAlloc(p as *const c_void, SIZE, MEM_COMMIT, PAGE_READWRITE) } as usize, p);
        for k in (0..SIZE / PAGE).step_by(2) {
            unsafe { std::ptr::write_volatile((p + k * PAGE) as *mut u8, 1) };
        }
        let mut map = Map::default();
        let walked = walk(p, p + 2 * SIZE, &mut Pages::new(), &mut map);
        // the same memory as one of the physical pages' homes: by its address first
        let mut home = Map { homes: vec![(p, 2 * SIZE)], ..Map::default() };
        walk(p, p + 2 * SIZE, &mut Pages::new(), &mut home);
        unsafe { VirtualFree(p as *mut c_void, 0, MEM_RELEASE) };

        assert_eq!(walked, 2, "the committed half and the reserved half");
        let b = map.classes[PRIVATE_MID];
        assert_eq!((b.committed, b.regions), (SIZE, 1));
        assert!(b.resident >= SIZE / 2 && b.resident < SIZE, "{} KiB resident of {} KiB committed, {} KiB touched", b.resident >> 10, SIZE >> 10, SIZE >> 11);
        assert_eq!(map.by_type, [SIZE, 0, 0]);
        assert_eq!(map.classes.iter().map(|c| c.committed).sum::<usize>(), SIZE, "in no other class");
        assert_eq!(map.allocs.len(), 1);
        let a = &map.allocs[0];
        assert_eq!((a.base, a.end, a.bytes.committed, a.bytes.resident), (p, p + 2 * SIZE, SIZE, b.resident), "one allocation, with its reserved half");
        assert_eq!(home.classes[HOMES].committed, SIZE);
        assert!(home.allocs.is_empty() && home.classes[PRIVATE_MID].committed == 0);
    }
}
