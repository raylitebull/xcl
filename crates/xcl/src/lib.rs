//! XCL — Xbox Series X|S → Windows compatibility layer.
//!
//! See `docs/specs/2026-11-20-xcl-design.md` for the design.

// Function → address casts are the whole point of an import resolver.
#![allow(function_casts_as_integer)]

#[macro_use]
pub mod log;
pub mod manifest;
pub(crate) mod cli;
pub mod config;
pub mod disasm;
pub mod eh;
pub mod gpu;
pub mod heap_count;
pub mod mem_map;
pub mod ram_budget;
pub mod hle;
pub mod hooks;
pub mod host;
pub mod insn;
pub mod inspect;
pub mod launcher;
pub mod loader;
pub mod modules;
pub mod pe;
pub mod preflight;
pub mod report;
pub mod resolver;
pub mod rpf;
mod rpf_probe;
pub mod shaders;
#[cfg(test)]
mod source_hash;
pub mod state;
pub mod stubs;
pub mod tls;
pub mod util;
pub mod vfs;
pub mod vfs_hooks;

/// XCL's own allocations go through the counters of `heap_count` (`XCL_HEAP_COUNT=1`).
#[global_allocator]
static HEAP: heap_count::Counting = heap_count::Counting;
