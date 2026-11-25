//! The compact form: each distinct block of a program lowered once, in a loop over a table.

use super::cfg::{Cfg, Term};
use super::lower::{ext_encoding, LResult, Lw, VAR_BUDGET, VAR_SCC};
use super::lower_mem::VAR_TMP;
use super::module::{Block, Pred, Value};
use std::collections::{BTreeSet, HashMap};

const ALL_SGPRS_MASK: u128 = (1u128 << 106) - 1;
const SCC_LIVE_BIT: u128 = 1u128 << VAR_SCC;

fn is_pure_nop(name: &str) -> bool {
    matches!(
        name,
        "S_NOP"
            | "V_NOP"
            | "S_WAITCNT"
            | "S_WAITCNT_VSCNT"
            | "S_WAITCNT_VMCNT"
            | "S_WAITCNT_LGKMCNT"
            | "S_WAITCNT_EXPCNT"
            | "S_WAITCNT_DEPCTR"
            | "S_INST_PREFETCH"
            | "S_CLAUSE"
    )
}

/// Conservative `(use_mask, def_mask)` over `s0..s105` (bits `0..106`) and `SCC` (bit `VAR_SCC`).
/// Any register not provably unconditionally overwritten is omitted from `def_mask`; any unknown
/// or indexed SGPR read sets all bits in `use_mask`.
fn inst_sreg_scc_use_def(d: &crate::gpu::rdna2::Decoded) -> (u128, u128) {
    use crate::gpu::rdna2::{field_value, register_scale, Variant};
    if d.inst.name.starts_with("S_MOVREL") {
        return (ALL_SGPRS_MASK, 0);
    }
    let cond_write = d.inst.name == "S_CMOVK_I32";
    let (mut u, mut def) = (0u128, 0u128);
    for o in d.ienc.operands {
        if o.ty == "OPR_SSRC_SPECIAL_SCC" {
            if o.input {
                u |= SCC_LIVE_BIT;
            }
            if o.output {
                def |= SCC_LIVE_BIT;
            }
            continue;
        }
        let Some(f) = o.field else { continue };
        if f == "LITERAL" || o.implicit {
            continue;
        }
        if !["SREG", "SSRC", "SDST", "OPR_SRC", "SMEM_OFFSET"].iter().any(|k| o.ty.contains(k)) {
            continue;
        }
        if f == "SRC0" && d.variant != Variant::Plain {
            if d.variant == Variant::Sdwa {
                let ext = d.words.get(1).copied().unwrap_or(0);
                if (ext >> 23) & 1 == 1 {
                    let code = ext & 0xFF;
                    if code < 106 {
                        u |= 1u128 << code;
                    } else if code == 253 {
                        u |= SCC_LIVE_BIT;
                    }
                }
            }
            continue;
        }
        let Some(raw) = field_value(d.enc, &d.words, f) else { continue };
        let v = raw * register_scale(f);
        if v == 253 && o.input {
            u |= SCC_LIVE_BIT;
            continue;
        }
        if v >= 106 {
            continue;
        }
        let n = (o.size as u64).div_ceil(32).max(1);
        for r in v..(v + n).min(106) {
            if o.input || cond_write {
                u |= 1u128 << r;
            }
            if o.output && !cond_write {
                def |= 1u128 << r;
            }
        }
    }
    (u, def)
}

fn compute_live_out(cfg: &Cfg, mblocks: &[MicroBlock], end_live: u128) -> Vec<u128> {
    let n = mblocks.len();
    let mut b_use = vec![0u128; n];
    let mut b_def = vec![0u128; n];
    for (idx, blk) in mblocks.iter().enumerate() {
        let (mut u, mut d) = (0u128, 0u128);
        for &i in &blk.insts {
            let (iu, id) = inst_sreg_scc_use_def(&cfg.insts[i]);
            u |= iu & !d;
            d |= id;
        }
        if let Term::Cond(ci, ..) = blk.term {
            let (iu, _) = inst_sreg_scc_use_def(&cfg.insts[ci]);
            u |= iu & !d;
        }
        b_use[idx] = u;
        b_def[idx] = d;
    }
    let mut live_in = vec![0u128; n];
    let mut live_out = vec![0u128; n];
    loop {
        let mut changed = false;
        for idx in (0..n).rev() {
            let out = match mblocks[idx].term {
                Term::End => end_live,
                Term::Jump(t) => if t < n { live_in[t] } else { end_live },
                Term::Cond(_, t, f) => {
                    let lt = if t < n { live_in[t] } else { end_live };
                    let lf = if f < n { live_in[f] } else { end_live };
                    lt | lf
                }
            };
            let inp = b_use[idx] | (out & !b_def[idx]);
            if out != live_out[idx] || inp != live_in[idx] {
                live_out[idx] = out;
                live_in[idx] = inp;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    live_out
}

/// If `blk` ends with a 3-instruction far-jump address computation (`s_getpc_b64 s[r:r+1]`,
/// `s_{add,sub}_u32 s[r]`, `s_{addc,subb}_u32 s[r+1]`) feeding a resolved `s_setpc_b64`,
/// return the destination SGPR base `r` (`< 105`) so dead far-jump veneers can be stripped.
fn resolved_far_jump_reg(cfg: &Cfg, blk: &MicroBlock) -> Option<u64> {
    if !matches!(blk.term, Term::Jump(_)) || blk.insts.len() < 3 {
        return None;
    }
    let len = blk.insts.len();
    let (i0, i1, i2) = (blk.insts[len - 3], blk.insts[len - 2], blk.insts[len - 1]);
    if i1 != i0 + 1 || i2 != i1 + 1 {
        return None;
    }
    let set = cfg.insts.get(i2 + 1)?;
    if set.inst.name != "S_SETPC_B64" || set.far_target.is_none() {
        return None;
    }
    let (get, lo, hi) = (&cfg.insts[i0], &cfg.insts[i1], &cfg.insts[i2]);
    if get.inst.name != "S_GETPC_B64" {
        return None;
    }
    if !matches!((lo.inst.name, hi.inst.name), ("S_ADD_U32", "S_ADDC_U32") | ("S_SUB_U32", "S_SUBB_U32")) {
        return None;
    }
    let r = crate::gpu::rdna2::field_value(get.enc, &get.words, "SDST")?;
    (r < 105).then_some(r)
}

/// One virtual basic block in a compact plan (either a whole guest block or a split micro-block
/// isolating a repeated heavy memory instruction so its lowering is emitted once instead of once
/// per surrounding ALU context).
pub struct MicroBlock {
    pub pc: u64,
    pub insts: Vec<usize>,
    pub term: Term,
    pub is_header: bool,
}

/// How the compact table encodes each micro-block's successor target(s).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NextEnc {
    /// Pair of relative `(taken_off, not_taken_off)` indexed in `xcl_steps`.
    Steps,
    /// `not_taken = b + 1`; `field` holds the absolute `taken` block number directly.
    Inline,
    /// `not_taken = b + 1`; `field == 0` means `taken = b + 1`, else `xcl_targets[field - 1]`.
    TargetPool,
}

pub struct Plan {
    /// the reachable blocks in address order (same length as `mblocks`)
    pub order: Vec<usize>,
    /// virtual blocks executed by the compact loop (targets in `term` are indices into `mblocks`)
    pub mblocks: Vec<MicroBlock>,
    /// the class of each
    pub class: Vec<u32>,
    /// a block of each class (a block number)
    pub rep: Vec<usize>,
    /// where each block goes: (branch taken or jump, branch not taken), as block numbers
    pub next: Vec<(u32, u32)>,
    /// the distinct pairs of offsets from a block to its two successors, sorted
    pub steps: Vec<(u32, u32)>,
    /// unique non-fallthrough taken targets for `NextEnc::TargetPool`
    pub targets: Vec<u32>,
    /// successor table encoding mode
    pub next_enc: NextEnc,
    /// an entry holds the block's taken successor itself (true for `Inline` and `TargetPool`)
    pub inline: bool,
    /// per-block parameterized 32-bit literal when the block has exactly one parameterizable literal
    pub blk_lit: Vec<Option<u32>>,
    /// whether `blk_lit` parameterization is enabled for this plan
    pub use_lit: bool,
    /// whether `S_GETPC_B64` block-PC parameterization is enabled for this plan
    pub use_pc: bool,
    /// union of live-out `(s0..s105 | SCC)` bitmasks across all micro-blocks in each class
    pub class_live_out: Vec<u128>,
}

/// Most 32-bit words of tables a program may need to take the form by itself.
const MAX_TABLE_WORDS: usize = 32768;

impl Plan {
    /// Bits of a table entry: the class, and the taken successor or the index of the pair of offsets.
    fn widths(&self) -> (u32, u32) {
        let pb = match self.next_enc {
            NextEnc::Steps => bits(self.steps.len()),
            NextEnc::Inline => bits(self.mblocks.len()),
            NextEnc::TargetPool => bits(self.targets.len() + 1),
        };
        (bits(self.rep.len()), pb)
    }

    /// 32-bit words of the tables.
    pub fn words(&self) -> usize {
        let (cb, pb) = self.widths();
        let extra = match self.next_enc {
            NextEnc::Steps => self.steps.len() * 2,
            NextEnc::Inline => 0,
            NextEnc::TargetPool => self.targets.len(),
        };
        (self.mblocks.len() * (cb + pb) as usize).div_ceil(32) + 1 + extra
    }
}

/// A consumed atomic result prevents cutting a dispatch into independent pipelines.
/// Consider a compact whole pipeline for branch-heavy split-wave compute shaders instead.
pub(super) fn atomic_candidate(cfg: &Cfg) -> bool {
    cfg.rpo.len() >= 250 && cfg.rpo.iter().flat_map(|&b| &cfg.blocks[b].insts).any(|&i| {
        let d = &cfg.insts[i];
        d.inst.name == "BUFFER_ATOMIC_ADD" && crate::gpu::rdna2::field_value(d.enc, &d.words, "GLC") == Some(1)
    })
}

pub fn automatic_plan(cfg: &Cfg, lw: &Lw, split_compute: bool) -> Option<Plan> {
    let large = large(cfg);
    if !large && !(split_compute && atomic_candidate(cfg)) { return None; }
    let p = plan(cfg, lw)?;
    if p.words() > MAX_TABLE_WORDS { return None; }
    let insts = |n: usize| p.mblocks[n].insts.len();
    let all: usize = (0..p.mblocks.len()).map(insts).sum();
    let once: usize = p.rep.iter().map(|&n| insts(n)).sum();
    let repeated = if large { once * 10 < all * 3 } else { p.rep.len() * 4 <= p.order.len() };
    repeated.then_some(p)
}

/// Whether to consider compaction based on instruction count alone.
pub fn large(cfg: &Cfg) -> bool {
    cfg.rpo.iter().map(|&b| cfg.blocks[b].insts.len()).sum::<usize>() >= 20_000
}

pub const END: u32 = u32::MAX;

fn is_heavy_mem(d: &crate::gpu::rdna2::Decoded) -> bool {
    matches!(
        d.enc.name,
        "ENC_MUBUF" | "ENC_MTBUF" | "ENC_SMEM" | "ENC_MIMG" | "ENC_FLAT" | "ENC_FLAT_GLBL" | "ENC_FLAT_SCRATCH"
    ) && !matches!(
        d.inst.name,
        "S_DCACHE_INV"
            | "S_DCACHE_WB"
            | "S_DCACHE_DISCARD"
            | "S_DCACHE_DISCARD_X2"
            | "S_GL1_INV"
            | "S_MEMTIME"
            | "S_MEMREALTIME"
    )
}

/// Whether a micro-block has exactly one parameterizable 32-bit literal instruction.
fn single_param_lit(cfg: &Cfg, blk: &MicroBlock) -> Option<u32> {
    let mut found = None;
    for &i in &blk.insts {
        let d = &cfg.insts[i];
        if let Some(lit) = d.literal {
            if d.inst.name == "S_SETREG_IMM32_B32" || d.enc.name == "ENC_MIMG" || found.is_some() {
                return None;
            }
            found = Some(lit);
        }
    }
    found
}

/// Sorts the reachable blocks into classes the lowering would emit the same for.
/// On large shaders, repeated heavy memory instructions (appearing `>= 4` times) are split
/// into standalone micro-blocks so each unique buffer/SMEM/image lowering is emitted once
/// instead of once per surrounding ALU context.
pub fn plan(cfg: &Cfg, lw: &Lw) -> Option<Plan> {
    let mut orig_order = cfg.rpo.clone();
    orig_order.sort_by_key(|&b| (cfg.blocks[b].pc, b));

    // Count exact heavy-memory instruction occurrences across reachable blocks.
    let mut mem_counts: HashMap<(Vec<u32>, Option<u32>, Option<[u32; 4]>), usize> = HashMap::new();
    let allow_split = orig_order.len() >= 64;
    if allow_split {
        for &b in &orig_order {
            for &i in &cfg.blocks[b].insts {
                let d = &cfg.insts[i];
                if is_heavy_mem(d) {
                    let sampler = lw.static_samplers.get(&d.pc).copied();
                    *mem_counts.entry((d.words.clone(), d.literal, sampler)).or_insert(0) += 1;
                }
            }
        }
    }

    // Pass 1: split blocks around repeated heavy memory ops (`>= 4` occurrences).
    let mut segs_per_orig: Vec<(usize, Vec<Vec<usize>>)> = Vec::with_capacity(orig_order.len());
    let mut orig_to_mb = vec![END; cfg.blocks.len()];
    let mut mb_cursor = 0u32;
    for &b in &orig_order {
        let blk = &cfg.blocks[b];
        let mut segs: Vec<Vec<usize>> = Vec::new();
        if allow_split && blk.insts.len() > 1 {
            let mut cur: Vec<usize> = Vec::new();
            for &i in &blk.insts {
                let d = &cfg.insts[i];
                let isolate = is_heavy_mem(d)
                    && mem_counts
                        .get(&(d.words.clone(), d.literal, lw.static_samplers.get(&d.pc).copied()))
                        .copied()
                        .unwrap_or(0)
                        >= 2;
                if isolate {
                    if !cur.is_empty() {
                        segs.push(std::mem::take(&mut cur));
                    }
                    segs.push(vec![i]);
                } else {
                    cur.push(i);
                }
            }
            if !cur.is_empty() {
                segs.push(cur);
            }
        }
        if segs.is_empty() {
            segs.push(blk.insts.clone());
        }
        orig_to_mb[b] = mb_cursor;
        mb_cursor += segs.len() as u32;
        segs_per_orig.push((b, segs));
    }

    // Pass 2: materialize micro-blocks with remapped successor targets (stripping pure no-ops).
    let mut order: Vec<usize> = Vec::with_capacity(mb_cursor as usize);
    let mut mblocks: Vec<MicroBlock> = Vec::with_capacity(mb_cursor as usize);
    for (orig_b, segs) in segs_per_orig {
        let blk = &cfg.blocks[orig_b];
        let is_hdr = cfg.headers.contains(&orig_b);
        let n_segs = segs.len();
        for (s, raw_insts) in segs.into_iter().enumerate() {
            let my_id = mblocks.len();
            let pc = raw_insts.first().map_or(blk.pc, |&i| cfg.insts[i].pc);
            let insts = if allow_split {
                raw_insts.into_iter().filter(|&i| !is_pure_nop(cfg.insts[i].inst.name)).collect()
            } else {
                raw_insts
            };
            let term = if s + 1 < n_segs {
                Term::Jump(my_id + 1)
            } else {
                match blk.term {
                    Term::End => Term::End,
                    Term::Jump(t) => Term::Jump(orig_to_mb[t] as usize),
                    Term::Cond(ci, t, f) => Term::Cond(ci, orig_to_mb[t] as usize, orig_to_mb[f] as usize),
                }
            };
            order.push(orig_b);
            mblocks.push(MicroBlock { pc, insts, term, is_header: s == 0 && is_hdr });
        }
    }

    // Pass 3 (large shaders): strip dead far-jump veneers (`s_getpc_b64 + s_add_u32 + s_addc_u32`
    // feeding a resolved `s_setpc_b64` when `s[r:r+1]` and `SCC` are dead at the jump target) and
    // bypass empty non-header jump blocks so neither `xcl_blocks` nor `FUNCTION` pays for them.
    let end_live = if lw.res.is_some() { ALL_SGPRS_MASK | SCC_LIVE_BIT } else { 0 };
    if allow_split {
        let live_out = compute_live_out(cfg, &mblocks, end_live);
        for (idx, blk) in mblocks.iter_mut().enumerate() {
            if let Some(r) = resolved_far_jump_reg(cfg, blk) {
                let clobber = (1u128 << r) | (1u128 << (r + 1)) | SCC_LIVE_BIT;
                if (live_out[idx] & clobber) == 0 {
                    let new_len = blk.insts.len() - 3;
                    blk.insts.truncate(new_len);
                }
            }
        }
        // Bypass empty non-header jump micro-blocks (entry 0 stays pinned at 0).
        // Preserve `falls` (`Term::Cond` fallthrough `f == idx + 1`) by not bypassing a
        // fallthrough block whose jump goes anywhere other than `cur + 1`.
        let n_mb = mblocks.len();
        let mut is_fallthrough = vec![false; n_mb];
        for (idx, blk) in mblocks.iter().enumerate() {
            if let Term::Cond(_, _, f) = blk.term {
                if f == idx + 1 && f < n_mb {
                    is_fallthrough[f] = true;
                }
            }
        }
        let mut redir: Vec<usize> = (0..n_mb).collect();
        for start in 1..n_mb {
            let mut cur = start;
            let mut ft = is_fallthrough[start];
            for _ in 0..64 {
                if cur == 0 || mblocks[cur].is_header || !mblocks[cur].insts.is_empty() {
                    break;
                }
                match mblocks[cur].term {
                    Term::Jump(t) if t < n_mb && t != cur => {
                        if ft && t != cur + 1 {
                            break;
                        }
                        cur = t;
                        ft |= is_fallthrough[cur];
                    }
                    _ => break,
                }
            }
            redir[start] = cur;
        }
        for blk in &mut mblocks {
            match &mut blk.term {
                Term::Jump(t) if *t < n_mb => *t = redir[*t],
                Term::Cond(_, t, f) => {
                    if *t < n_mb {
                        *t = redir[*t];
                    }
                    if *f < n_mb {
                        *f = redir[*f];
                    }
                }
                _ => {}
            }
            if let Term::Cond(_, t, f) = blk.term {
                if t == f {
                    blk.term = Term::Jump(t);
                }
            }
        }
        // Compact out bypassed/unreachable micro-blocks.
        let mut reach = vec![false; n_mb];
        let mut stack = vec![0usize];
        reach[0] = true;
        while let Some(u) = stack.pop() {
            match mblocks[u].term {
                Term::Jump(t) if t < n_mb && !reach[t] => {
                    reach[t] = true;
                    stack.push(t);
                }
                Term::Cond(_, t, f) => {
                    if t < n_mb && !reach[t] {
                        reach[t] = true;
                        stack.push(t);
                    }
                    if f < n_mb && !reach[f] {
                        reach[f] = true;
                        stack.push(f);
                    }
                }
                _ => {}
            }
        }
        let mut remap = vec![END as usize; n_mb];
        let mut new_order = Vec::with_capacity(n_mb);
        let mut new_mblocks = Vec::with_capacity(n_mb);
        for (i, (orig_b, blk)) in order.into_iter().zip(mblocks.into_iter()).enumerate() {
            if reach[i] {
                remap[i] = new_mblocks.len();
                new_order.push(orig_b);
                new_mblocks.push(blk);
            }
        }
        for blk in &mut new_mblocks {
            match &mut blk.term {
                Term::Jump(t) if *t < n_mb => *t = remap[*t],
                Term::Cond(_, t, f) => {
                    if *t < n_mb {
                        *t = remap[*t];
                    }
                    if *f < n_mb {
                        *f = remap[*f];
                    }
                }
                _ => {}
            }
        }
        order = new_order;
        mblocks = new_mblocks;
    }

    let final_live_out = if allow_split {
        compute_live_out(cfg, &mblocks, end_live)
    } else {
        vec![ALL_SGPRS_MASK | SCC_LIVE_BIT; mblocks.len()]
    };

    let build = |param_lit: bool, param_pc: bool| {
        let mut ids: HashMap<(Vec<u64>, &'static str), u32> = HashMap::new();
        let (mut class, mut rep, mut next, mut blk_lit) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut falls = true;
        for (n, blk) in mblocks.iter().enumerate() {
            let p_lit = if param_lit { single_param_lit(cfg, blk) } else { None };
            blk_lit.push(p_lit);
            let mut key = vec![blk.is_header as u64, p_lit.is_some() as u64];
            for &i in &blk.insts {
                let d = &cfg.insts[i];
                let mask_lit = p_lit.is_some() && d.literal.is_some();
                let words = if mask_lit { &d.words[..d.words.len() - 1] } else { &d.words[..] };
                key.push(d.words.len() as u64);
                key.extend(words.iter().map(|&w| w as u64));
                if mask_lit {
                    key.push(u64::MAX - 1);
                } else {
                    key.push(d.literal.map_or(u64::MAX, |l| l as u64));
                }
                let sampler = lw.static_samplers.get(&d.pc);
                key.push(sampler.is_some() as u64);
                key.extend(sampler.into_iter().flatten().map(|&w| w as u64));
                // what `Lw::write_mask` asks of this address, for any register a mask can go to
                if let Some(&l) = lw.whole_live.get(&d.pc) {
                    let sdwa = ext_encoding(d).and_then(|e| crate::gpu::rdna2::field_value(e, &d.words, "SDST"));
                    let to = crate::gpu::rdna2::sgpr_uses(d).1.into_iter().chain([106]).chain(sdwa);
                    key.extend(to.filter(|&r| r < 126).map(|r| (l >> r) as u64 & 3));
                }
                if d.inst.name == "S_GETPC_B64" {
                    key.push(if param_pc { d.pc.wrapping_sub(blk.pc) } else { d.pc });
                }
            }
            let (end, to) = match blk.term {
                Term::End => ("end", (END, END)),
                Term::Jump(t) => ("jump", (t as u32, t as u32)),
                Term::Cond(i, t, f) => {
                    falls &= f == n + 1;
                    (cfg.insts[i].inst.name, (t as u32, f as u32))
                }
            };
            let fresh = ids.len() as u32;
            let c = *ids.entry((key, end)).or_insert(fresh);
            if c == fresh {
                rep.push(n);
            }
            class.push(c);
            next.push(to);
        }
        (class, rep, next, blk_lit, falls)
    };
    let base = build(false, false);
    let has_getpc = allow_split && mblocks.iter().any(|blk| blk.insts.iter().any(|&i| cfg.insts[i].inst.name == "S_GETPC_B64"));
    // Parameterize single-literal blocks (and S_GETPC_B64 when present) on large shaders only
    // when the collapsed switch arms outweigh the packed lookup tables (`~2 B/block` per table).
    let (use_lit, use_pc, (class, rep, next, blk_lit, falls)) = if allow_split {
        let cand = build(true, has_getpc);
        let saved_both = base.1.len().saturating_sub(cand.1.len());
        let table_cost = mblocks.len() * if has_getpc { 4 } else { 2 };
        if saved_both >= 64 && saved_both * 60 > table_cost {
            let only_lit = if has_getpc { build(true, false) } else { cand.clone() };
            let saved_pc = only_lit.1.len().saturating_sub(cand.1.len());
            if has_getpc && saved_pc >= 64 && saved_pc * 60 > mblocks.len() * 2 {
                (true, true, cand)
            } else if base.1.len().saturating_sub(only_lit.1.len()) >= 64
                && base.1.len().saturating_sub(only_lit.1.len()) * 60 > mblocks.len() * 2
            {
                (true, false, only_lit)
            } else {
                (false, false, base)
            }
        } else {
            (false, false, base)
        }
    } else {
        (false, false, base)
    };
    let mut class_live_out = vec![0u128; rep.len()];
    for (mb_idx, &c) in class.iter().enumerate() {
        class_live_out[c as usize] |= final_live_out[mb_idx];
    }
    let steps: BTreeSet<(u32, u32)> = next.iter().enumerate().map(|(b, &to)| offsets(b, to)).collect();
    let targets: Vec<u32> = if falls {
        let set: BTreeSet<u32> = next
            .iter()
            .enumerate()
            .filter_map(|(b, &(t, _))| (t != END && t != b as u32 + 1).then_some(t))
            .collect();
        set.into_iter().collect()
    } else {
        Vec::new()
    };
    let mut p = Plan {
        order,
        mblocks,
        class,
        rep,
        next,
        steps: steps.into_iter().collect(),
        targets,
        next_enc: NextEnc::Steps,
        inline: false,
        blk_lit,
        use_lit,
        use_pc,
        class_live_out,
    };
    let mut best_enc = NextEnc::Steps;
    let mut best_words = p.words();
    if falls {
        for cand_enc in [NextEnc::Inline, NextEnc::TargetPool] {
            if cand_enc == NextEnc::TargetPool && p.targets.is_empty() {
                continue;
            }
            p.next_enc = cand_enc;
            let (cb, pb) = p.widths();
            if cb + pb <= 32 && p.words() < best_words {
                best_words = p.words();
                best_enc = cand_enc;
            }
        }
    }
    p.next_enc = best_enc;
    p.inline = matches!(best_enc, NextEnc::Inline | NextEnc::TargetPool);
    let (cb, pb) = p.widths();
    (cb + pb <= 32).then_some(p)
}

/// A block's successors as offsets from its own number.
fn offsets(b: usize, (t, f): (u32, u32)) -> (u32, u32) {
    if t == END { (0, 0) } else { (t.wrapping_sub(b as u32), f.wrapping_sub(b as u32)) }
}

fn bits(n: usize) -> u32 {
    (usize::BITS - n.saturating_sub(1).leading_zeros()).max(1)
}

/// A constant table of 32-bit words (emitted via LLVM `CST_CODE_DATA` without interning
/// each element as an individual `CST_CODE_INTEGER` constant).
fn table(lw: &mut Lw, name: &str, words: &[u32]) -> Value {
    let i32 = lw.s.m.i32();
    let at = lw.s.m.array(words.len() as u64, i32);
    let init = lw.s.m.const_data_i32(words);
    lw.s.m.global(name, at, 0, Some(init), true, 4)
}

fn word(lw: &mut Lw, table: Value, i: Value) -> Value {
    let z = lw.c(0);
    let p = lw.s.m.gep(table, &[z, i]);
    lw.s.m.load(p, 4)
}

fn slot(lw: &mut Lw, var: u32) -> Value {
    let Lw { s, init, .. } = lw;
    init.slot(s, var)
}

/// Default class threshold and inner group size for `Nest::Auto`.
pub const AUTO_NEST_MIN_CLASSES: usize = 512;
pub const AUTO_NEST_GROUP_SIZE: usize = 32;

/// Dispatch nesting policy for the direct DXIL compact form (`XCL_DIRECT_COMPACT_NEST`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nest {
    /// Single flat `switch(class)` with one shared loop latch.
    Flat,
    /// Two-level nested switch (groups of [`AUTO_NEST_GROUP_SIZE`]) once class count reaches
    /// [`AUTO_NEST_MIN_CLASSES`], flat below that.
    Auto,
    /// Two-level nested switch with explicit power-of-two inner group size whenever `classes > p`.
    Fixed(usize),
}

impl Nest {
    /// Resolves the inner group size for a program with `classes` distinct block classes,
    /// returning `None` for flat dispatch.
    pub fn group_size(self, classes: usize) -> Option<usize> {
        match self {
            Nest::Flat => None,
            Nest::Auto => (classes >= AUTO_NEST_MIN_CLASSES).then_some(AUTO_NEST_GROUP_SIZE),
            Nest::Fixed(p) => (p >= 2 && p.is_power_of_two() && classes > p).then_some(p),
        }
    }

    /// Non-default cache tag folded into the direct-DXIL translated cache identity.
    pub fn cache_tag(self) -> Option<String> {
        match self {
            Nest::Auto => None,
            Nest::Flat => Some("xcl-direct-compact-nest-0".into()),
            Nest::Fixed(p) => Some(format!("xcl-direct-compact-nest-{p}")),
        }
    }
}

/// `XCL_DIRECT_COMPACT_NEST`: `auto` or unset nests `>= 512` classes into groups of 32;
/// `0` / `off` keeps a flat switch; a power of two in `2..=65536` fixes the inner group size.
pub fn nest_mode() -> Nest {
    parse_nest(std::env::var("XCL_DIRECT_COMPACT_NEST").ok().as_deref())
}

pub(crate) fn parse_nest(val: Option<&str>) -> Nest {
    let Some(raw) = val.map(str::trim) else { return Nest::Auto };
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Nest::Auto;
    }
    if raw == "0" || raw.eq_ignore_ascii_case("off") {
        return Nest::Flat;
    }
    match raw.parse::<usize>() {
        Ok(p) if (2..=65536).contains(&p) && p.is_power_of_two() => Nest::Fixed(p),
        _ => Nest::Auto,
    }
}

/// Lowers the program as one loop from `head` to `exit` that switches to each block's class.
pub fn emit(lw: &mut Lw, cfg: &Cfg, p: &Plan, head: Block, exit: Block) -> LResult<()> {
    emit_with_nest(lw, cfg, p, head, exit, nest_mode())
}

/// Like [`emit`], with an explicit [`Nest`] policy.
pub fn emit_with_nest(lw: &mut Lw, cfg: &Cfg, p: &Plan, head: Block, exit: Block, nest: Nest) -> LResult<()> {
    let n = p.order.len();
    let (cb, pb) = p.widths();
    let width = cb + pb;
    let mut packed = vec![0u32; (n * width as usize).div_ceil(32) + 1];
    for (b, &to) in p.next.iter().enumerate() {
        let field = match (p.next_enc, to.0) {
            (NextEnc::Inline, END) => 0,
            (NextEnc::Inline, t) => t,
            (NextEnc::TargetPool, END) => 0,
            (NextEnc::TargetPool, t) if t == b as u32 + 1 => 0,
            (NextEnc::TargetPool, t) => p.targets.binary_search(&t).expect("target in pool") as u32 + 1,
            (NextEnc::Steps, _) => p.steps.binary_search(&offsets(b, to)).expect("a pair of the plan") as u32,
        };
        let e = (p.class[b] | field << cb) as u64;
        let at = b * width as usize;
        packed[at / 32] |= (e << (at % 32)) as u32;
        packed[at / 32 + 1] |= (e << (at % 32) >> 32) as u32;
    }
    let blocks = table(lw, "xcl_blocks", &packed);
    let steps: Vec<u32> = p.steps.iter().flat_map(|&(t, f)| [t, f]).collect();
    let steps = (p.next_enc == NextEnc::Steps).then(|| table(lw, "xcl_steps", &steps));
    let target_pool = (p.next_enc == NextEnc::TargetPool).then(|| table(lw, "xcl_targets", &p.targets));
    let pc_table = p.use_pc.then(|| {
        let max_pc_dw = p.mblocks.iter().map(|blk| (blk.pc >> 2) as usize).max().unwrap_or(0);
        let pc_w = bits(max_pc_dw + 1).min(32);
        let mut pc_packed = vec![0u32; (n * pc_w as usize).div_ceil(32) + 1];
        for (b, blk) in p.mblocks.iter().enumerate() {
            let v = (blk.pc >> 2) as u64;
            let at = b * pc_w as usize;
            pc_packed[at / 32] |= (v << (at % 32)) as u32;
            pc_packed[at / 32 + 1] |= (v << (at % 32) >> 32) as u32;
        }
        (table(lw, "xcl_pc", &pc_packed), pc_w)
    });
    let lit_table = p.use_lit.then(|| {
        // Deduplicate unique 32-bit literals into a pool and bit-pack pool indices per block.
        let mut pool: Vec<u32> = vec![0];
        let mut idx_map: HashMap<u32, u32> = HashMap::from([(0, 0)]);
        let mut blk_idx: Vec<u32> = Vec::with_capacity(n);
        for &opt in &p.blk_lit {
            let val = opt.unwrap_or(0);
            let idx = *idx_map.entry(val).or_insert_with(|| {
                let id = pool.len() as u32;
                pool.push(val);
                id
            });
            blk_idx.push(idx);
        }
        let lit_w = bits(pool.len()).min(32);
        let mut idx_packed = vec![0u32; (n * lit_w as usize).div_ceil(32) + 1];
        for (b, &id) in blk_idx.iter().enumerate() {
            let v = id as u64;
            let at = b * lit_w as usize;
            idx_packed[at / 32] |= (v << (at % 32)) as u32;
            idx_packed[at / 32 + 1] |= (v << (at % 32) >> 32) as u32;
        }
        (table(lw, "xcl_lit_idx", &idx_packed), table(lw, "xcl_lit_pool", &pool), lit_w)
    });

    let (at, to) = (lw.tmp_var(), lw.tmp_var());
    let (at, to, budget) = (slot(lw, at), slot(lw, to), slot(lw, VAR_BUDGET));
    lw.s.m.set_block(head);
    let b = lw.s.m.load(at, 4);
    let w = lw.c(width);
    let bit = lw.mul(b, w);
    let i = lw.lshrc(bit, 5);
    let sh = lw.andc(bit, 31);
    let lo = word(lw, blocks, i);
    let i1 = lw.addc(i, 1);
    let hi = word(lw, blocks, i1);
    let lo = lw.lshr(lo, sh);
    let hi = lw.shlc(hi, 1);
    let k31 = lw.c(31);
    let up = lw.sub(k31, sh);
    let hi = lw.shl(hi, up);
    let e = lw.or(lo, hi);
    let class = lw.andc(e, (1u32 << cb) - 1);
    let field = lw.lshrc(e, cb);
    let field = lw.andc(field, (1u32 << pb) - 1);
    let blk_pc = pc_table.map(|(pc_tbl, pc_w)| {
        let pm = if pc_w == 32 {
            word(lw, pc_tbl, b)
        } else {
            let pw = lw.c(pc_w);
            let pbit = lw.mul(b, pw);
            let pi = lw.lshrc(pbit, 5);
            let psh = lw.andc(pbit, 31);
            let plo = word(lw, pc_tbl, pi);
            let pi1 = lw.addc(pi, 1);
            let phi = word(lw, pc_tbl, pi1);
            let plo = lw.lshr(plo, psh);
            let phi = lw.shlc(phi, 1);
            let pup = lw.sub(k31, psh);
            let phi = lw.shl(phi, pup);
            let pe = lw.or(plo, phi);
            lw.andc(pe, (1u32 << pc_w) - 1)
        };
        lw.shlc(pm, 2)
    });
    let blk_lit_val = lit_table.map(|(idx_tbl, pool_tbl, lit_w)| {
        let id = if lit_w == 32 {
            word(lw, idx_tbl, b)
        } else {
            let lw_c = lw.c(lit_w);
            let lbit = lw.mul(b, lw_c);
            let li = lw.lshrc(lbit, 5);
            let lsh = lw.andc(lbit, 31);
            let llo = word(lw, idx_tbl, li);
            let li1 = lw.addc(li, 1);
            let lhi = word(lw, idx_tbl, li1);
            let llo = lw.lshr(llo, lsh);
            let lhi = lw.shlc(lhi, 1);
            let lup = lw.sub(k31, lsh);
            let lhi = lw.shl(lhi, lup);
            let le = lw.or(llo, lhi);
            lw.andc(le, (1u32 << lit_w) - 1)
        };
        word(lw, pool_tbl, id)
    });
    let (taken, not_taken) = if let Some(steps) = steps {
        let pi = lw.shlc(field, 1);
        let step_t = word(lw, steps, pi);
        let pi1 = lw.addc(pi, 1);
        let step_f = word(lw, steps, pi1);
        (lw.add(b, step_t), lw.add(b, step_f))
    } else if let Some(target_tbl) = target_pool {
        let next_b = lw.addc(b, 1);
        let is_fall = lw.cmpc(Pred::Eq, field, 0);
        let one = lw.c(1);
        let tidx = lw.sub(field, one);
        let z = lw.c(0);
        let safe_tidx = lw.sel(is_fall, z, tidx);
        let pooled = word(lw, target_tbl, safe_tidx);
        (lw.sel(is_fall, next_b, pooled), next_b)
    } else {
        (field, lw.addc(b, 1))
    };
    let cases: Vec<Block> = p.rep.iter().map(|_| lw.block(Some(0))).collect();
    let latch = lw.block(Some(0));
    let group_size = nest.group_size(p.rep.len());
    let group_latches: Vec<Block> = if let Some(gs) = group_size {
        let groups = p.rep.len().div_ceil(gs);
        let shift = gs.trailing_zeros();
        let entries: Vec<Block> = (0..groups).map(|_| lw.block(Some(0))).collect();
        let latches: Vec<Block> = (0..groups).map(|_| lw.block(Some(0))).collect();
        let grp = lw.lshrc(class, shift);
        let outer_arms: Vec<(Value, Block)> = entries.iter().enumerate().skip(1).map(|(g, &blk)| (lw.c(g as u32), blk)).collect();
        lw.s.m.switch(grp, entries[0], &outer_arms);
        for (g, &entry_blk) in entries.iter().enumerate() {
            let lo = g * gs;
            let hi = (lo + gs).min(p.rep.len());
            lw.s.m.set_block(entry_blk);
            if hi - lo == 1 {
                lw.s.m.br(cases[lo]);
            } else {
                let inner_arms: Vec<(Value, Block)> = (lo + 1..hi).map(|k| (lw.c(k as u32), cases[k])).collect();
                lw.s.m.switch(class, cases[lo], &inner_arms);
            }
        }
        latches
    } else {
        let arms: Vec<(Value, Block)> = cases.iter().enumerate().skip(1).map(|(k, &c)| (lw.c(k as u32), c)).collect();
        lw.s.m.switch(class, cases[0], &arms);
        Vec::new()
    };
    for (k, (&c, &r)) in cases.iter().zip(&p.rep).enumerate() {
        let target_latch = group_size.map_or(latch, |gs| group_latches[k / gs]);
        lw.s.m.set_block(c);
        lw.init.mem = Some(c);
        lw.ssa.begin_log();
        let blk = &p.mblocks[r];
        if blk.is_header {
            // a loop head spends the loop budget
            let left = lw.s.m.load(budget, 4);
            let none = lw.cmpc(Pred::Eq, left, 0);
            let (out, body) = (lw.block(Some(1)), lw.block(Some(1)));
            lw.cond_br(none, out, body);
            lw.s.m.set_block(out);
            let end = lw.c(END);
            lw.s.m.store(to, end, 4);
            lw.s.m.br(target_latch);
            lw.s.m.set_block(body);
            let one = lw.c(1);
            let l = lw.sub(left, one);
            lw.s.m.store(budget, l, 4);
        }
        lw.compact_blk_pc = blk_pc.map(|v| (v, blk.pc));
        lw.compact_lit = if p.blk_lit[r].is_some() { blk_lit_val } else { None };
        for &i in &blk.insts {
            lw.inst(&cfg.insts[i])?;
        }
        lw.compact_blk_pc = None;
        lw.compact_lit = None;
        let next = match blk.term {
            Term::End => lw.c(END),
            Term::Jump(_) => taken,
            Term::Cond(i, ..) => {
                let c = lw.branch_cond(&cfg.insts[i])?;
                lw.sel(c, taken, not_taken)
            }
        };
        let live_out = p.class_live_out[k];
        for var in lw.ssa.end_log().into_iter().filter(|&v| v < VAR_TMP) {
            if (var < 106 || var == VAR_SCC) && (live_out & (1u128 << var)) == 0 {
                continue;
            }
            let (v, p) = (lw.rd(var), slot(lw, var));
            lw.s.m.store(p, v, 4);
        }
        lw.s.m.store(to, next, 4);
        lw.s.m.br(target_latch);
    }
    if group_size.is_some() {
        // Each group latch performs the loop-exit check rather than forwarding with an empty
        // unconditional branch, so driver CFG simplification preserves the two-level merge fan-in
        // during SSA promotion without adding extra runtime instructions.
        for &gl in &group_latches {
            lw.s.m.set_block(gl);
            let nxt = lw.s.m.load(to, 4);
            let stop = lw.cmpc(Pred::UGe, nxt, n as u32);
            lw.s.m.cond_br(stop, exit, latch);
        }
        // the epilogue's reads load in exit
        lw.init.mem = Some(exit);
        lw.s.m.set_block(latch);
        let next = lw.s.m.load(to, 4);
        lw.s.m.store(at, next, 4);
        lw.br(head);
        lw.seal(head);
    } else {
        // the epilogue's reads load here
        lw.init.mem = Some(latch);
        lw.s.m.set_block(latch);
        let next = lw.s.m.load(to, 4);
        let stop = lw.cmpc(Pred::UGe, next, n as u32);
        let cont = lw.block(Some(1));
        lw.cond_br(stop, exit, cont);
        lw.s.m.set_block(cont);
        lw.s.m.store(at, next, 4);
        lw.br(head);
        lw.seal(head);
    }
    Ok(())
}
