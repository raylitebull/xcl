//! Cheap cleanup of the entry function before it is written: unreachable
//! blocks emptied, trivial phis replaced by their single value, common
//! subexpressions merged (pure instructions along the dominator tree, wave
//! intrinsics within a block), dead instructions removed.
//!
//! The translator emits each guest instruction's per-lane expansion as is
//! (repeated lane-bit / ballot / extract sequences, operands of discarded
//! results); this pass removes the redundancy without changing what any
//! remaining instruction computes.

use std::collections::HashMap;

use super::{BinOp, Block, CastOp, FnAttrs, Module, Op, Pred, Ty, Value};

impl Op {
    /// Calls `f` on every value operand.
    pub(super) fn for_each_operand(&self, mut f: impl FnMut(Value)) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                f(*a);
                f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::Call(_, args) => args.iter().copied().for_each(f),
            Op::Phi(inc) => inc.iter().for_each(|p| f(p.0)),
            Op::Gep(p, idx) => {
                f(*p);
                idx.iter().copied().for_each(f)
            }
            Op::Switch(v, _, cases) => {
                f(*v);
                cases.iter().for_each(|c| f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }

    /// Rewrites every value operand through `f`.
    pub(super) fn map_operands(&mut self, mut f: impl FnMut(Value) -> Value) {
        match self {
            Op::Bin(_, a, b, _) | Op::Cmp(_, a, b) | Op::Store(a, b, _) | Op::AtomicRmw(_, a, b) => {
                *a = f(*a);
                *b = f(*b)
            }
            Op::Cast(_, v, _) | Op::Extract(v, _) | Op::Load(v, _) | Op::Alloca(_, v, _) | Op::CondBr(v, ..) => *v = f(*v),
            Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
                *a = f(*a);
                *b = f(*b);
                *c = f(*c)
            }
            Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Op::Phi(inc) => inc.iter_mut().for_each(|p| p.0 = f(p.0)),
            Op::Gep(p, idx) => {
                *p = f(*p);
                idx.iter_mut().for_each(|v| *v = f(*v))
            }
            Op::Switch(v, _, cases) => {
                *v = f(*v);
                cases.iter_mut().for_each(|c| c.0 = f(c.0))
            }
            Op::Br(_) | Op::Ret | Op::Unreachable => {}
        }
    }
}

/// How an instruction may be merged with an identical one.
#[derive(PartialEq)]
enum Cse {
    No,
    /// pure: any dominating copy will do
    Dom,
    /// a wave intrinsic: same result only under the same active lanes (same block)
    Block,
}

impl Module {
    fn wave_call(&self, f: u32) -> bool {
        let n = &self.funcs[f as usize].name;
        n.starts_with("dx.op.wave") || n.starts_with("dx.op.quad")
    }

    fn is_const_global_ptr(&self, v: Value) -> bool {
        match v {
            Value::Global(g) => self.globals[g as usize].constant,
            Value::Inst(i) => match &self.insts[i as usize].op {
                Op::Gep(base, _) => self.is_const_global_ptr(*base),
                _ => false,
            },
            _ => false,
        }
    }

    fn cse_kind(&self, op: &Op) -> Cse {
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) => Cse::Dom,
            Op::Load(p, _) if self.is_const_global_ptr(*p) => Cse::Dom,
            Op::Phi(_) if cfg!(test) && std::env::var_os("XCL_SAMPLE_PHI_CSE").is_some() => Cse::Block,
            Op::Call(f, _) if self.funcs[*f as usize].attrs == FnAttrs::ReadNone => {
                let name = &self.funcs[*f as usize].name;
                // AMD RDNA4 (`amdxc64.dll`) crashes on null deref if a raw `createHandleFromHeap`
                // handle is shared across distinct `annotateHandle` resource kinds or across basic
                // blocks, or if `annotateHandle` / `getDimensions` crosses basic blocks.
                if name.starts_with("dx.op.createHandleFromHeap") {
                    Cse::No
                } else if name.starts_with("dx.op.annotateHandle") || name.starts_with("dx.op.getDimensions") {
                    Cse::Block
                } else {
                    Cse::Dom
                }
            }
            Op::Call(f, _) if self.funcs[*f as usize].name.starts_with("dx.op.getDimensions") => Cse::Block,
            Op::Call(f, _) if self.wave_call(*f) && !self.funcs[*f as usize].name.starts_with("dx.op.waveIsFirstLane") => Cse::Block,
            _ => Cse::No,
        }
    }

    /// Cheap algebraic & constant simplification on already-resolved operands.
    fn simplify_op(&mut self, op: &Op, ty: Ty) -> Option<Value> {
        // Keep 64-bit instructions intact so Int64Ops / Doubles shader flags stay consistent.
        if matches!(self.types[ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
            return None;
        }
        match *op {
            Op::Select(c, t, f) => {
                if t == f {
                    return Some(t);
                }
                match self.const_u64(c) {
                    Some(1) => Some(t),
                    Some(0) => Some(f),
                    _ => None,
                }
            }
            Op::Bin(bop, a, b, 0) => {
                let w = self.int_width(ty)?;
                // avoid dropping 64-bit ops that keep Int64Ops shader flags consistent
                if w > 32 {
                    return None;
                }
                let mask = (1u64 << w) - 1;
                let (ca, cb) = (self.const_u64(a), self.const_u64(b));
                if let (Some(x), Some(y)) = (ca, cb) {
                    let r = match bop {
                        BinOp::Add => Some(x.wrapping_add(y)),
                        BinOp::Sub => Some(x.wrapping_sub(y)),
                        BinOp::Mul => Some(x.wrapping_mul(y)),
                        BinOp::And => Some(x & y),
                        BinOp::Or => Some(x | y),
                        BinOp::Xor => Some(x ^ y),
                        BinOp::Shl if y < w as u64 => Some(x << y),
                        BinOp::LShr if y < w as u64 => Some(x >> y),
                        BinOp::AShr if y < w as u64 => Some((((x << (64 - w)) as i64) >> (64 - w + y as u32)) as u64),
                        BinOp::UDiv if y != 0 => Some(x / y),
                        BinOp::URem if y != 0 => Some(x % y),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r & mask));
                    }
                }
                match (bop, cb) {
                    (BinOp::Add | BinOp::Sub | BinOp::Or | BinOp::Xor | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::And, Some(m)) if m == mask => return Some(a),
                    (BinOp::Or, Some(m)) if m == mask => return Some(b),
                    (BinOp::And | BinOp::Mul, Some(0)) => return Some(b),
                    (BinOp::Mul, Some(1)) => return Some(a),
                    _ => {}
                }
                match (bop, ca) {
                    (BinOp::Add | BinOp::Or | BinOp::Xor, Some(0)) => return Some(b),
                    (BinOp::And, Some(m)) if m == mask => return Some(b),
                    (BinOp::Or, Some(m)) if m == mask => return Some(a),
                    (BinOp::And | BinOp::Mul | BinOp::Shl | BinOp::LShr | BinOp::AShr, Some(0)) => return Some(a),
                    (BinOp::Mul, Some(1)) => return Some(b),
                    _ => {}
                }
                if a == b {
                    match bop {
                        BinOp::And | BinOp::Or => return Some(a),
                        BinOp::Sub | BinOp::Xor => return Some(self.const_int(ty, 0)),
                        _ => {}
                    }
                }
                match bop {
                    BinOp::Xor => {
                        for (cand, other) in [(a, b), (b, a)] {
                            if let Value::Inst(j) = cand {
                                if let Op::Bin(BinOp::Xor, x, y, 0) = self.insts[j as usize].op {
                                    if other == y {
                                        return Some(x);
                                    }
                                    if other == x {
                                        return Some(y);
                                    }
                                }
                            }
                        }
                    }
                    BinOp::And => {
                        if let Some((v, m)) = cb.map(|m| (a, m)).or_else(|| ca.map(|m| (b, m))) {
                            if let Value::Inst(j) = v {
                                match self.insts[j as usize].op {
                                    Op::Bin(BinOp::LShr, _, sh, 0) => {
                                        if let Some(s) = self.const_u64(sh).filter(|&s| s < w as u64) {
                                            let known = mask >> s;
                                            if known & m == known {
                                                return Some(v);
                                            }
                                            if known & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Bin(BinOp::Shl, _, sh, 0) => {
                                        if let Some(s) = self.const_u64(sh).filter(|&s| s < w as u64) {
                                            let known = (mask << s) & mask;
                                            if known & m == known {
                                                return Some(v);
                                            }
                                            if known & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Bin(BinOp::And, x, y, 0) => {
                                        if let Some(c1) = self.const_u64(y).or_else(|| self.const_u64(x)) {
                                            if c1 & m == c1 {
                                                return Some(v);
                                            }
                                            if c1 & m == 0 {
                                                return Some(self.const_int(ty, 0));
                                            }
                                        }
                                    }
                                    Op::Cast(CastOp::ZExt, inner, _) if self.int_width(self.value_ty(inner)) == Some(1) => {
                                        return Some(if m & 1 != 0 { v } else { self.const_int(ty, 0) });
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    BinOp::Or => {
                        if let Some((v, m)) = cb.map(|m| (a, m)).or_else(|| ca.map(|m| (b, m))) {
                            if let Value::Inst(j) = v {
                                if let Op::Bin(BinOp::Or, x, y, 0) = self.insts[j as usize].op {
                                    if let Some(c1) = self.const_u64(y).or_else(|| self.const_u64(x)) {
                                        if c1 | m == c1 {
                                            return Some(v);
                                        }
                                        if (c1 | m) == mask {
                                            return Some(self.const_int(ty, mask));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
                None
            }
            Op::Cmp(p, a, b) => {
                let at = self.value_ty(a);
                let w = self.int_width(at)?;
                if w > 32 {
                    return None;
                }
                if let (Some(x), Some(y)) = (self.const_u64(a), self.const_u64(b)) {
                    let sx = |v: u64| ((v << (64 - w)) as i64) >> (64 - w);
                    let r = match p {
                        Pred::Eq => Some(x == y),
                        Pred::Ne => Some(x != y),
                        Pred::UGt => Some(x > y),
                        Pred::UGe => Some(x >= y),
                        Pred::ULt => Some(x < y),
                        Pred::ULe => Some(x <= y),
                        Pred::SGt => Some(sx(x) > sx(y)),
                        Pred::SGe => Some(sx(x) >= sx(y)),
                        Pred::SLt => Some(sx(x) < sx(y)),
                        Pred::SLe => Some(sx(x) <= sx(y)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                if a == b {
                    let r = match p {
                        Pred::Eq | Pred::ULe | Pred::UGe | Pred::SLe | Pred::SGe => Some(true),
                        Pred::Ne | Pred::ULt | Pred::UGt | Pred::SLt | Pred::SGt => Some(false),
                        _ => None,
                    };
                    if let Some(r) = r {
                        return Some(self.const_int(ty, r as u64));
                    }
                }
                None
            }
            Op::Cast(CastOp::BitCast, Value::Inst(j), to) => {
                if matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Int(64) | super::TypeDef::Double) {
                    return None;
                }
                if let Op::Cast(CastOp::BitCast, inner, _) = self.insts[j as usize].op {
                    if self.value_ty(inner) == to {
                        return Some(inner);
                    }
                }
                None
            }
            Op::Cast(cop @ (CastOp::ZExt | CastOp::SExt | CastOp::Trunc), v, to) => {
                let sw = self.int_width(self.value_ty(v))?;
                let dw = self.int_width(to)?;
                if sw > 32 || dw > 32 {
                    return None;
                }
                let x = self.const_u64(v)?;
                let r = match cop {
                    CastOp::ZExt | CastOp::Trunc => x & ((1u64 << dw) - 1),
                    CastOp::SExt => ((((x << (64 - sw)) as i64) >> (64 - sw)) as u64) & ((1u64 << dw) - 1),
                    _ => unreachable!(),
                };
                Some(self.const_int(to, r))
            }
            _ => None,
        }
    }

    /// Whether an unused instruction can go. 64-bit values stay: helpers already set
    /// the Int64Ops / Doubles / DoubleExtensions flags for them.
    fn removable(&self, op: &Op, ty: Ty, wide: &[Ty]) -> bool {
        if wide.contains(&ty) {
            return false;
        }
        let mut w = false;
        op.for_each_operand(|v| w |= wide.contains(&self.value_ty(v)));
        if w {
            return false;
        }
        match op {
            Op::Bin(..) | Op::Cast(..) | Op::Cmp(..) | Op::Select(..) | Op::Extract(..) | Op::Gep(..) | Op::Phi(_) | Op::Load(..) | Op::Alloca(..) => true,
            // only calls whose presence sets no shader flag (the flags are collected while
            // building and must match the code: no wave ops, f64 math, typed / raw loads)
            Op::Call(f, _) => {
                let d = &self.funcs[*f as usize];
                matches!(d.attrs, FnAttrs::ReadNone | FnAttrs::ReadOnly)
                    && !d.name.ends_with(".f64")
                    && ["dx.op.unary.", "dx.op.binary.", "dx.op.tertiary.", "dx.op.loadInput.", "dx.op.annotateHandle", "dx.op.createHandleFromBinding", "dx.op.cbufferLoadLegacy."]
                        .iter()
                        .any(|p| d.name.starts_with(p))
            }
            _ => false,
        }
    }

    /// Whether every non-phi operand of the reachable instructions is written before its
    /// use and has a value (what the writer checks).
    fn uses_well_formed(&self, order: &[u32]) -> bool {
        let mut at = vec![usize::MAX; self.insts.len()];
        let mut k = 0;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                at[i as usize] = k;
                k += 1;
            }
        }
        let mut ok = true;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                if matches!(op, Op::Phi(_)) {
                    continue;
                }
                op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        ok &= at[j as usize] < at[i as usize] && !matches!(self.types[self.insts[j as usize].ty.0 as usize], super::TypeDef::Void);
                    }
                });
            }
        }
        ok
    }

    /// Immediate dominators of the blocks in `order` (reverse postorder), by
    /// Cooper, Harvey and Kennedy's iteration ("A Simple, Fast Dominance Algorithm").
    pub(super) fn idoms(&self, order: &[u32]) -> Vec<Option<usize>> {
        let n = self.blocks.len();
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b as usize] = k;
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &b in order {
            for s in self.successors(b as usize) {
                preds[s.0 as usize].push(b as usize);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut new: Option<usize> = None;
                for &p in &preds[b] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(mut x) => {
                            let mut y = p;
                            while x != y {
                                while rpo[x] > rpo[y] {
                                    x = idom[x].unwrap();
                                }
                                while rpo[y] > rpo[x] {
                                    y = idom[y].unwrap();
                                }
                            }
                            x
                        }
                    });
                }
                if new.is_some() && idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Replace phis whose incoming values are only themselves and one other value. Repeat
    /// because resolving one phi may make another trivial, including across loop backedges.
    fn simplify_phis(&self, order: &[u32], repl: &mut [Option<Value>]) {
        loop {
            let mut changed = false;
            for &b in order {
                for &i in &self.blocks[b as usize] {
                    let i = i as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let Op::Phi(inc) = &self.insts[i].op else { break };
                    let me = Value::Inst(i as u32);
                    let mut same: Option<Value> = None;
                    let mut trivial = true;
                    for &(v, _) in inc {
                        let v = resolve(repl, v);
                        if v == me || Some(v) == same {
                            continue;
                        }
                        if same.is_some() {
                            trivial = false;
                            break;
                        }
                        same = Some(v);
                    }
                    if let (true, Some(v)) = (trivial, same) {
                        repl[i] = Some(v);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    /// Constants established by the control-flow edge that reaches each block. Facts are
    /// intersected at joins, so a value is kept only when every incoming edge proves the same
    /// value. Bounded to 16 RPO passes so giant shaders stay strictly linear.
    fn branch_facts(&self, order: &[u32]) -> Vec<HashMap<Value, u64>> {
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
        for &b in order {
            for s in self.successors(b as usize) {
                if !preds[s.0 as usize].contains(&(b as usize)) {
                    preds[s.0 as usize].push(b as usize);
                }
            }
        }
        let mut facts: Vec<HashMap<Value, u64>> = vec![HashMap::new(); self.blocks.len()];
        let max_iters = order.len().min(16);
        for _ in 0..=max_iters {
            let mut changed = false;
            for &b in order.iter().skip(1) {
                let b = b as usize;
                let mut incoming: Option<HashMap<Value, u64>> = None;
                for &p in &preds[b] {
                    let mut edge = facts[p].clone();
                    if let Some(&term) = self.blocks[p].last() {
                        match &self.insts[term as usize].op {
                            Op::CondBr(v, t, f) if t != f => {
                                if t.0 as usize == b {
                                    edge.insert(*v, 1);
                                } else if f.0 as usize == b {
                                    edge.insert(*v, 0);
                                }
                            }
                            Op::Switch(v, default, cases) if default.0 as usize != b => {
                                let matching: Vec<Value> = cases.iter().filter(|(_, t)| t.0 as usize == b).map(|(c, _)| *c).collect();
                                if let [case] = matching.as_slice() {
                                    if let Some(value) = self.const_u64(*case) {
                                        edge.insert(*v, value);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    incoming = Some(match incoming {
                        None => edge,
                        Some(mut common) => {
                            common.retain(|v, value| edge.get(v) == Some(value));
                            common
                        }
                    });
                }
                let next = incoming.unwrap_or_default();
                if facts[b] != next {
                    facts[b] = next;
                    changed = true;
                }
            }
            if !changed {
                return facts;
            }
        }
        vec![HashMap::new(); self.blocks.len()]
    }

    fn is_hoistable_readnone_call(&self, f: u32, args: &[Value], in_entry: &[bool]) -> bool {
        let fd = &self.funcs[f as usize];
        if fd.attrs != FnAttrs::ReadNone || fd.name.ends_with(".f64") {
            return false;
        }
        // Never hoist `createHandleFromHeap`, `annotateHandle`, or `getDimensions` across blocks
        // (AMD RDNA4 `amdxc64` requires heap handles / annotations / dimensions to stay block-local).
        let ok_name = [
            "dx.op.groupId.",
            "dx.op.threadId.",
            "dx.op.threadIdInGroup.",
            "dx.op.flattenedThreadIdInGroup.",
            "dx.op.createHandleFromBinding",
        ]
        .iter()
        .any(|p| fd.name.starts_with(p));
        if !ok_name {
            return false;
        }
        args.iter().all(|a| match *a {
            Value::Const(_) => true,
            Value::Inst(j) => in_entry.get(j as usize).copied().unwrap_or(false),
            _ => false,
        })
    }

    /// Hoists duplicate entry-available `ReadNone` intrinsics (such as `dx.op.groupId` in
    /// compact switch arms, or `createHandleFromHeap` + `annotateHandle` pairs) into entry
    /// block 0 so every block shares a single dominating definition.
    fn hoist_entry_readnone(&mut self, order: &[u32], repl: &mut Vec<Option<Value>>) -> bool {
        if order.is_empty() || self.blocks[0].is_empty() {
            return false;
        }
        let mut any_changed = false;
        // Two passes: pass 0 hoists `createHandleFromHeap` / `groupId`, pass 1 hoists
        // `annotateHandle` whose handle argument was just hoisted into entry block 0.
        for _pass in 0..2 {
            let mut in_entry = vec![false; self.insts.len()];
            for &i in &self.blocks[0] {
                if repl[i as usize].is_none() {
                    in_entry[i as usize] = true;
                }
            }
            let mut counts: HashMap<(u32, Vec<Value>), usize> = HashMap::new();
            for &b in order {
                for k in 0..self.blocks[b as usize].len() {
                    let i = self.blocks[b as usize][k] as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let (f, raw_args) = match &self.insts[i].op {
                        Op::Call(f, args) => (*f, args.clone()),
                        _ => continue,
                    };
                    let args: Vec<Value> = raw_args.into_iter().map(|a| resolve(repl, a)).collect();
                    if self.is_hoistable_readnone_call(f, &args, &in_entry) {
                        *counts.entry((f, args)).or_insert(0) += 1;
                    }
                }
            }
            let mut hoisted: HashMap<(u32, Vec<Value>), Value> = HashMap::new();
            // Reuse any call already in entry block 0 first.
            for k in 0..self.blocks[0].len() {
                let i = self.blocks[0][k] as usize;
                if repl[i].is_some() {
                    continue;
                }
                let (f, raw_args) = match &self.insts[i].op {
                    Op::Call(f, args) => (*f, args.clone()),
                    _ => continue,
                };
                let args: Vec<Value> = raw_args.into_iter().map(|a| resolve(repl, a)).collect();
                let key = (f, args);
                if counts.get(&key).copied().unwrap_or(0) >= 2 {
                    hoisted.entry(key).or_insert(Value::Inst(i as u32));
                }
            }
            let mut changed = false;
            let term = if self.blocks[0].last().is_some_and(|&t| {
                matches!(
                    self.insts[t as usize].op,
                    Op::Br(_) | Op::CondBr(..) | Op::Switch(..) | Op::Ret | Op::Unreachable
                )
            }) {
                self.blocks[0].pop()
            } else {
                None
            };
            for &b in order {
                for k in 0..self.blocks[b as usize].len() {
                    let i = self.blocks[b as usize][k] as usize;
                    if repl[i].is_some() {
                        continue;
                    }
                    let (f, raw_args, ty) = match &self.insts[i].op {
                        Op::Call(f, args) => (*f, args.clone(), self.insts[i].ty),
                        _ => continue,
                    };
                    let args: Vec<Value> = raw_args.into_iter().map(|a| resolve(repl, a)).collect();
                    if !self.is_hoistable_readnone_call(f, &args, &in_entry) {
                        continue;
                    }
                    let key = (f, args);
                    if counts.get(&key).copied().unwrap_or(0) < 2 {
                        continue;
                    }
                    let entry_val = *hoisted.entry(key.clone()).or_insert_with(|| {
                        let id = self.insts.len() as u32;
                        self.insts.push(super::InstDef { op: Op::Call(key.0, key.1), ty });
                        self.def_block.push(Block(0));
                        self.blocks[0].push(id);
                        repl.push(None);
                        Value::Inst(id)
                    });
                    if entry_val != Value::Inst(i as u32) {
                        repl[i] = Some(entry_val);
                        changed = true;
                    }
                }
            }
            if let Some(t) = term {
                self.blocks[0].push(t);
            }
            if !changed {
                break;
            }
            any_changed = true;
        }
        any_changed
    }

    /// Intra-block store-to-load forwarding and dead-store elimination for non-escaping scalar
    /// `alloca` slots (primarily the register slots of `compact.rs`), plus removal of stores to
    /// scalar allocas that have zero surviving loads anywhere in the function.
    fn opt_local_allocas(&mut self, order: &[u32], repl: &mut [Option<Value>]) -> bool {
        let mut scalar_allocas: Vec<bool> = vec![false; self.insts.len()];
        let mut any_alloca = false;
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let idx = i as usize;
                if repl[idx].is_some() {
                    continue;
                }
                if let Op::Alloca(ty, n, _) = self.insts[idx].op {
                    if self.const_u64(n) == Some(1)
                        && !matches!(
                            self.types[ty.0 as usize],
                            super::TypeDef::Array(..) | super::TypeDef::Struct(..) | super::TypeDef::Int(64) | super::TypeDef::Double
                        )
                    {
                        scalar_allocas[idx] = true;
                        any_alloca = true;
                    }
                }
            }
        }
        if !any_alloca {
            return false;
        }
        // Disqualify any alloca whose address escapes (used anywhere except as ptr of Load/Store).
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let idx = i as usize;
                if repl[idx].is_some() {
                    continue;
                }
                match &self.insts[idx].op {
                    Op::Load(p, _) => {
                        let _ = p; // pointer operand is valid
                    }
                    Op::Store(_p, v, _) => {
                        if let Value::Inst(j) = *v {
                            scalar_allocas[j as usize] = false;
                        }
                    }
                    other => other.for_each_operand(|v| {
                        if let Value::Inst(j) = v {
                            scalar_allocas[j as usize] = false;
                        }
                    }),
                }
            }
        }
        if !scalar_allocas.iter().any(|&x| x) {
            return false;
        }

        let mut dead_store = vec![false; self.insts.len()];
        let mut loaded = vec![false; self.insts.len()];
        let mut changed = false;

        for &b in order {
            let mut known: HashMap<u32, Value> = HashMap::new();
            let mut last_store: HashMap<u32, usize> = HashMap::new();
            let len = self.blocks[b as usize].len();
            for k in 0..len {
                let i = self.blocks[b as usize][k] as usize;
                if repl[i].is_some() {
                    continue;
                }
                match self.insts[i].op {
                    Op::Load(Value::Inst(a), _) if scalar_allocas[a as usize] => {
                        if let Some(&v) = known.get(&a) {
                            let v = resolve(repl, v);
                            if self.value_ty(v) == self.insts[i].ty {
                                repl[i] = Some(v);
                                changed = true;
                                continue;
                            }
                        }
                        loaded[a as usize] = true;
                        last_store.remove(&a);
                        known.insert(a, Value::Inst(i as u32));
                    }
                    Op::Store(Value::Inst(a), v, _) if scalar_allocas[a as usize] => {
                        let v = resolve(repl, v);
                        if known.get(&a).copied().map(|kv| resolve(repl, kv)) == Some(v) {
                            dead_store[i] = true;
                            changed = true;
                            continue;
                        }
                        if let Some(prev) = last_store.insert(a, i) {
                            dead_store[prev] = true;
                            changed = true;
                        }
                        known.insert(a, v);
                    }
                    _ => {}
                }
            }
        }

        // Any scalar alloca with zero surviving loads anywhere in the function needs no stores.
        for &b in order {
            for &i in &self.blocks[b as usize] {
                let idx = i as usize;
                if !dead_store[idx] {
                    if let Op::Store(Value::Inst(a), _, _) = self.insts[idx].op {
                        if scalar_allocas[a as usize] && !loaded[a as usize] {
                            dead_store[idx] = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        if changed {
            for &b in order {
                self.blocks[b as usize].retain(|&i| !dead_store[i as usize]);
            }
        }
        changed
    }

    /// Removes immediately redundant `dx.op.barrier` calls within a basic block when no
    /// memory-visible side effect occurred since an equally-strong or stronger barrier.
    fn elim_redundant_barriers(&mut self, order: &[u32]) -> bool {
        let mut removed = false;
        for &b in order {
            let mut active_mask: Option<u64> = None;
            let mut keep: Vec<u32> = Vec::with_capacity(self.blocks[b as usize].len());
            for &i in &self.blocks[b as usize] {
                let op = &self.insts[i as usize].op;
                match op {
                    Op::Call(f, args) if self.funcs[*f as usize].name == "dx.op.barrier" => {
                        let mode = args.get(1).and_then(|&v| self.const_u64(v));
                        match (active_mask, mode) {
                            (Some(prev), Some(cur)) if (cur & !prev) == 0 => {
                                removed = true;
                                continue;
                            }
                            (Some(prev), Some(cur)) => active_mask = Some(prev | cur),
                            (None, Some(cur)) => active_mask = Some(cur),
                            _ => active_mask = None,
                        }
                    }
                    Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => active_mask = None,
                    Op::Load(p, _) if !self.is_const_global_ptr(*p) => active_mask = None,
                    Op::Call(f, _) if self.funcs[*f as usize].attrs != FnAttrs::ReadNone => active_mask = None,
                    _ => {}
                }
                keep.push(i);
            }
            if keep.len() != self.blocks[b as usize].len() {
                self.blocks[b as usize] = keep;
            }
        }
        removed
    }

    /// Recursively derives implied constant facts from a known boolean value `v == val` (bounded
    /// depth so complex expressions stay strictly O(1)).
    fn record_fact(
        &self,
        v: Value,
        val: u64,
        depth: u8,
        facts: &mut HashMap<Value, u64>,
        undo: &mut Vec<(Value, Option<u64>)>,
    ) {
        if matches!(v, Value::Const(_)) {
            return;
        }
        if facts.get(&v) == Some(&val) {
            return;
        }
        let old = facts.insert(v, val);
        undo.push((v, old));
        if depth == 0 {
            return;
        }
        let Value::Inst(i) = v else { return };
        match self.insts[i as usize].op {
            Op::Bin(BinOp::And, a, b, 0) if val == 1 && self.int_width(self.insts[i as usize].ty) == Some(1) => {
                self.record_fact(a, 1, depth - 1, facts, undo);
                self.record_fact(b, 1, depth - 1, facts, undo);
            }
            Op::Bin(BinOp::Or, a, b, 0) if val == 0 && self.int_width(self.insts[i as usize].ty) == Some(1) => {
                self.record_fact(a, 0, depth - 1, facts, undo);
                self.record_fact(b, 0, depth - 1, facts, undo);
            }
            Op::Bin(BinOp::Xor, a, b, 0) if self.int_width(self.insts[i as usize].ty) == Some(1) => {
                if self.const_u64(b) == Some(1) {
                    self.record_fact(a, val ^ 1, depth - 1, facts, undo);
                } else if self.const_u64(a) == Some(1) {
                    self.record_fact(b, val ^ 1, depth - 1, facts, undo);
                }
            }
            Op::Cmp(Pred::Eq, a, b) if val == 1 => {
                if self.int_width(self.value_ty(a)).is_some_and(|w| w <= 32) {
                    if let Some(k) = self.const_u64(b) {
                        self.record_fact(a, k, depth - 1, facts, undo);
                    } else if let Some(k) = self.const_u64(a) {
                        self.record_fact(b, k, depth - 1, facts, undo);
                    }
                }
            }
            Op::Cmp(Pred::Ne, a, b) if val == 0 => {
                if self.int_width(self.value_ty(a)).is_some_and(|w| w <= 32) {
                    if let Some(k) = self.const_u64(b) {
                        self.record_fact(a, k, depth - 1, facts, undo);
                    } else if let Some(k) = self.const_u64(a) {
                        self.record_fact(b, k, depth - 1, facts, undo);
                    }
                }
            }
            _ => {}
        }
    }

    fn apply_facts_to_op(&mut self, op: &mut Op, facts: &HashMap<Value, u64>) -> bool {
        if facts.is_empty() {
            return false;
        }
        let mut changed = false;
        let lookup = |m: &mut Self, v: &mut Value, changed: &mut bool| {
            if let Some(&k) = facts.get(v) {
                let ty = m.value_ty(*v);
                if m.int_width(ty).is_some_and(|w| w <= 32) {
                    *v = m.const_int(ty, k);
                    *changed = true;
                }
            }
        };
        match op {
            Op::Select(c, t, f) => {
                lookup(self, c, &mut changed);
                lookup(self, t, &mut changed);
                lookup(self, f, &mut changed);
            }
            Op::CondBr(c, ..) => lookup(self, c, &mut changed),
            Op::Switch(v, ..) => lookup(self, v, &mut changed),
            Op::Bin(_, a, b, 0) | Op::Cmp(_, a, b) => {
                lookup(self, a, &mut changed);
                lookup(self, b, &mut changed);
            }
            Op::Cast(CastOp::ZExt | CastOp::SExt | CastOp::Trunc, v, _) => {
                lookup(self, v, &mut changed);
            }
            _ => {}
        }
        changed
    }

    /// Cleans up the entry function (see the module docs of `opt`). Values
    /// keep their identities; replaced ones simply lose their uses.
    ///
    /// `XCL_DXIL_NO_OPT=phi,cse,cfg,dce,merge,alloca,facts,hoist,barrier` (any subset, or `all`)
    /// skips passes, for bisecting.
    pub fn optimize(&mut self) {
        let off = std::env::var("XCL_DXIL_NO_OPT").unwrap_or_default();
        let skip = |p: &str| off.split(',').any(|x| x.trim() == p || x.trim() == "all");
        let order0: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
        // malformed IR (a use before its definition or of a void result) is left for the
        // writer to report, not cleaned away
        if !self.uses_well_formed(&order0) {
            return;
        }

        const MAX_OPT_ITERS: usize = 4;
        for _iter in 0..MAX_OPT_ITERS {
            let mut iter_changed = false;
            let mut order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let insts_before: usize = order.iter().map(|&b| self.blocks[b as usize].len()).sum();
            let blocks_before = order.len();

            // unreachable blocks: emptied (the writer drops them)
            let mut reach = vec![false; self.blocks.len()];
            for &b in &order {
                reach[b as usize] = true;
            }
            for (b, r) in reach.iter().enumerate() {
                if !r && !self.blocks[b].is_empty() {
                    self.blocks[b].clear();
                    iter_changed = true;
                }
            }
            for &b in &order {
                for k in 0..self.blocks[b as usize].len() {
                    let i = self.blocks[b as usize][k] as usize;
                    if let Op::Phi(inc) = &mut self.insts[i].op {
                        inc.retain(|p| reach[p.1 .0 as usize]);
                    }
                }
            }

            let mut repl: Vec<Option<Value>> = vec![None; self.insts.len()];

            // trivial phis (all incoming values the phi itself or one value), to a fixed point
            if !skip("phi") {
                self.simplify_phis(&order, &mut repl);
            }

            // intra-block scalar alloca store->load forwarding & dead-store elimination
            if !skip("alloca") && self.opt_local_allocas(&order, &mut repl) {
                iter_changed = true;
            }

            // hoist repeated constant-argument ReadNone intrinsics (e.g. dx.op.groupId) to entry
            if !skip("hoist") {
                if self.hoist_entry_readnone(&order, &mut repl) {
                    repl.resize(self.insts.len(), None);
                    iter_changed = true;
                }
            }

            // CSE + FACTS branch-condition propagation along the dominator tree
            let idom = self.idoms(&order);
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b in &order {
                for s in self.successors(b as usize) {
                    if !preds[s.0 as usize].contains(&(b as usize)) {
                        preds[s.0 as usize].push(b as usize);
                    }
                }
            }
            let mut kids: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b in order.iter().skip(1) {
                if let Some(d) = idom[b as usize] {
                    kids[d].push(b as usize);
                }
            }
            let use_facts = !skip("facts");
            let mut table: HashMap<(Op, Ty), Value> = HashMap::new();
            let mut undo: Vec<((Op, Ty), Option<Value>)> = Vec::new();
            let mut facts: HashMap<Value, u64> = HashMap::new();
            let mut fact_undo: Vec<(Value, Option<u64>)> = Vec::new();
            enum Step {
                Enter(usize),
                Leave(usize, usize),
            }
            let mut stack = if skip("cse") { Vec::new() } else { vec![Step::Enter(0)] };
            let mut cse_replaced = false;
            while let Some(step) = stack.pop() {
                let b = match step {
                    Step::Leave(mark, fmark) => {
                        while undo.len() > mark {
                            let (k, old) = undo.pop().unwrap();
                            match old {
                                Some(v) => table.insert(k, v),
                                None => table.remove(&k),
                            };
                        }
                        while fact_undo.len() > fmark {
                            let (k, old) = fact_undo.pop().unwrap();
                            match old {
                                Some(v) => facts.insert(k, v),
                                None => facts.remove(&k),
                            };
                        }
                        continue;
                    }
                    Step::Enter(b) => b,
                };
                stack.push(Step::Leave(undo.len(), fact_undo.len()));
                // Push dominating branch facts when `b` has a single CFG predecessor (`idom[b]`).
                if use_facts && preds[b].len() == 1 {
                    let p = preds[b][0];
                    if let Some(&term_i) = self.blocks[p].last() {
                        match self.insts[term_i as usize].op.clone() {
                            Op::CondBr(c, t, f) if t != f => {
                                let c = resolve(&mut repl, c);
                                if t.0 as usize == b {
                                    self.record_fact(c, 1, 4, &mut facts, &mut fact_undo);
                                } else if f.0 as usize == b {
                                    self.record_fact(c, 0, 4, &mut facts, &mut fact_undo);
                                }
                            }
                            Op::Switch(v, default, cases) if default.0 as usize != b => {
                                let matching: Vec<Value> = cases.iter().filter(|(_, t)| t.0 as usize == b).map(|(c, _)| *c).collect();
                                if let [case] = matching.as_slice() {
                                    if let Some(k) = self.const_u64(*case) {
                                        let v = resolve(&mut repl, v);
                                        self.record_fact(v, k, 2, &mut facts, &mut fact_undo);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let mut local: HashMap<(Op, Ty), Value> = HashMap::new();
                for k in 0..self.blocks[b].len() {
                    let i = self.blocks[b][k] as usize;
                    let mut op = std::mem::replace(&mut self.insts[i].op, Op::Ret);
                    op.map_operands(|v| resolve(&mut repl, v));
                    if use_facts && !facts.is_empty() && self.apply_facts_to_op(&mut op, &facts) {
                        cse_replaced = true;
                    }
                    // Collapse chained EXEC-predicated VGPR selects under the same condition in place:
                    // `select(c, t, select(c, _, f2)) -> select(c, t, f2)` and
                    // `select(c, select(c, t2, _), f) -> select(c, t2, f)`.
                    if let Op::Select(c, ref mut t, ref mut f) = op {
                        if let Value::Inst(j) = *f {
                            if let Op::Select(c2, _, f2) = self.insts[j as usize].op {
                                if c2 == c {
                                    *f = resolve(&mut repl, f2);
                                    cse_replaced = true;
                                }
                            }
                        }
                        if let Value::Inst(j) = *t {
                            if let Op::Select(c2, t2, _) = self.insts[j as usize].op {
                                if c2 == c {
                                    *t = resolve(&mut repl, t2);
                                    cse_replaced = true;
                                }
                            }
                        }
                    }
                    if repl[i].is_none() {
                        if let Some(s) = self.simplify_op(&op, self.insts[i].ty) {
                            repl[i] = Some(s);
                            cse_replaced = true;
                        } else {
                            let kind = self.cse_kind(&op);
                            if kind != Cse::No {
                                let key = (op.clone(), self.insts[i].ty);
                                let t = if kind == Cse::Dom { &mut table } else { &mut local };
                                if let Some(&v) = t.get(&key) {
                                    repl[i] = Some(v);
                                    cse_replaced = true;
                                } else {
                                    let old = t.insert(key.clone(), Value::Inst(i as u32));
                                    if kind == Cse::Dom {
                                        undo.push((key, old));
                                    }
                                }
                            } else if matches!(&op, Op::Call(f, _) if self.funcs[*f as usize].name == "dx.op.discard") {
                                // `dx.op.discard` demotes lanes mid-block, changing the active lane
                                // set for subsequent wave/quad intrinsics in the same basic block.
                                local.clear();
                            }
                        }
                    }
                    self.insts[i].op = op;
                }
                for &c in kids[b].iter().rev() {
                    stack.push(Step::Enter(c));
                }
            }
            if cse_replaced {
                iter_changed = true;
            }
            // second trivial-phi fixed point if CSE/simplification collapsed incoming values
            if cse_replaced && !skip("phi") {
                self.simplify_phis(&order, &mut repl);
            }
            // phis may name values defined later (back edges): resolve before CFG folding so
            // terminators see constants exposed by both CSE and the second trivial-phi pass.
            for &b in &order {
                for k in 0..self.blocks[b as usize].len() {
                    let i = self.blocks[b as usize][k] as usize;
                    self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
                }
            }
            // CSE, trivial-phi folding, or dominating branches can prove a terminator constant. Fold
            // it before DCE, then rebuild reachability and predecessor edges so unreachable blocks and
            // obsolete PHI edges are pruned cleanly while preserving loop backedges.
            let mut cfg_folded = false;
            if !skip("cfg") {
                let bfacts = self.branch_facts(&order);
                for &b in &order {
                    let Some(&i) = self.blocks[b as usize].last() else { continue };
                    let replacement = match self.insts[i as usize].op.clone() {
                        Op::CondBr(c, t, f) => match self.const_u64(c).or_else(|| bfacts[b as usize].get(&c).copied()) {
                            Some(0) => Some(Op::Br(f)),
                            Some(1) => Some(Op::Br(t)),
                            _ => None,
                        },
                        Op::Switch(v, default, cases) => self.const_u64(v).or_else(|| bfacts[b as usize].get(&v).copied()).map(|needle| {
                            let target = cases.iter().find_map(|(case, target)| (self.const_u64(*case) == Some(needle)).then_some(*target)).unwrap_or(default);
                            Op::Br(target)
                        }),
                        _ => None,
                    };
                    if let Some(op) = replacement {
                        self.insts[i as usize].op = op;
                        cfg_folded = true;
                        iter_changed = true;
                    }
                }
            }
            if cfg_folded {
                order = self.block_layout().0.into_iter().map(|b| b as u32).collect();
                let mut reach = vec![false; self.blocks.len()];
                let mut preds: Vec<Vec<u32>> = vec![Vec::new(); self.blocks.len()];
                for &b in &order {
                    reach[b as usize] = true;
                    for succ in self.successors(b as usize) {
                        if let Some(p) = preds.get_mut(succ.0 as usize) {
                            if !p.contains(&b) {
                                p.push(b);
                            }
                        }
                    }
                }
                for (b, r) in reach.iter().enumerate() {
                    if !r {
                        self.blocks[b].clear();
                    }
                }
                for &b in &order {
                    for &i in &self.blocks[b as usize] {
                        let Op::Phi(inc) = &mut self.insts[i as usize].op else { break };
                        inc.retain(|p| preds[b as usize].contains(&p.1 .0));
                    }
                }
                if !skip("phi") {
                    self.simplify_phis(&order, &mut repl);
                    for &b in &order {
                        for k in 0..self.blocks[b as usize].len() {
                            let i = self.blocks[b as usize][k] as usize;
                            self.insts[i].op.map_operands(|v| resolve(&mut repl, v));
                        }
                    }
                }
            }

            if !skip("barrier") && self.elim_redundant_barriers(&order) {
                iter_changed = true;
            }

            // dead instructions: live = side effects, terminators and their operands, transitively
            let (i64t, f64t) = self.existing_i64_f64();
            let wide: Vec<Ty> = i64t.into_iter().chain(f64t).collect();
            let mut live = vec![false; self.insts.len()];
            let mut work: Vec<u32> = Vec::new();
            for &b in &order {
                for &i in &self.blocks[b as usize] {
                    if repl[i as usize].is_none() && (skip("dce") || !self.removable(&self.insts[i as usize].op, self.insts[i as usize].ty, &wide)) {
                        live[i as usize] = true;
                        work.push(i);
                    }
                }
            }
            while let Some(i) = work.pop() {
                self.insts[i as usize].op.for_each_operand(|v| {
                    if let Value::Inst(j) = v {
                        if !live[j as usize] {
                            live[j as usize] = true;
                            work.push(j);
                        }
                    }
                });
            }
            for b in &mut self.blocks {
                b.retain(|&i| live[i as usize]);
            }

            if !skip("merge") && self.simplify_cfg() {
                iter_changed = true;
            }

            let order_after: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let insts_after: usize = order_after.iter().map(|&b| self.blocks[b as usize].len()).sum();
            debug_assert!(self.uses_well_formed(&order_after));
            if !iter_changed || (insts_after == insts_before && order_after.len() == blocks_before) {
                break;
            }
        }
    }

    /// Simplifies the CFG to a fixed point:
    /// 1. Collapses degenerate `CondBr(_, t, t)` and `Switch` cases matching the default target.
    /// 2. Bypasses empty trampoline blocks (`[Br(t)]`) whenever `t`'s phis allow it.
    /// 3. Merges single-predecessor blocks `a -> b` along unconditional branches.
    fn simplify_cfg(&mut self) -> bool {
        let mut ever_changed = false;
        let mut phi_repl: Vec<Option<Value>> = Vec::new();
        loop {
            let mut changed = false;
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();

            // Step 0: simplify degenerate terminators (`CondBr(_, t, t)`, `Switch` cases -> default).
            for &b_u32 in &order {
                let b = b_u32 as usize;
                let Some(&term_i) = self.blocks[b].last() else { continue };
                let b_blk = Block(b_u32);
                match &mut self.insts[term_i as usize].op {
                    Op::CondBr(_, t, f) if *t == *f => {
                        let target = *t;
                        self.insts[term_i as usize].op = Op::Br(target);
                        self.drop_phi_edges(target.0 as usize, b_blk, 1);
                        changed = true;
                    }
                    Op::Switch(_, def, cases) => {
                        let def = *def;
                        let before = cases.len();
                        cases.retain(|&(_, target)| target != def);
                        let dropped = before - cases.len();
                        let became_empty = cases.is_empty();
                        if dropped > 0 {
                            self.drop_phi_edges(def.0 as usize, b_blk, dropped);
                            changed = true;
                        }
                        if became_empty {
                            self.insts[term_i as usize].op = Op::Br(def);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }

            // Step 1: bypass empty trampoline blocks (`[Br(t)]`, `b != 0`, `t != b`).
            // `preds[x]` is maintained as a superset of live blocks whose terminator targets `x`.
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); self.blocks.len()];
            for &b_u32 in &order {
                let b = b_u32 as usize;
                for s in self.successors(b) {
                    let s = s.0 as usize;
                    if !preds[s].contains(&b) {
                        preds[s].push(b);
                    }
                }
            }
            for &b_u32 in &order {
                let b = b_u32 as usize;
                if b == 0 || self.blocks[b].len() != 1 {
                    continue;
                }
                let Op::Br(t_blk) = self.insts[self.blocks[b][0] as usize].op else { continue };
                let t = t_blk.0 as usize;
                if t == b || self.blocks[t].is_empty() {
                    continue;
                }
                let b_blk = Block(b_u32);
                let cand_preds = preds[b].clone();
                let target_has_phi = self.blocks[t]
                    .first()
                    .is_some_and(|&i| matches!(self.insts[i as usize].op, Op::Phi(_)));
                if target_has_phi {
                    // Bypassing a multi-edge trampoline into a phi block would duplicate phi
                    // operands for every phi in `t`; only bypass when `b` has a single incoming edge.
                    let total_edges: usize = cand_preds
                        .iter()
                        .filter(|&&p| p != b && !self.blocks[p].is_empty())
                        .map(|&p| self.count_term_edges(p, b_blk))
                        .sum();
                    if total_edges > 1 {
                        continue;
                    }
                }
                let mut all_redirected = true;
                let mut any_redirected = false;
                for p in cand_preds {
                    if p == b || self.blocks[p].is_empty() {
                        continue;
                    }
                    let edges = self.count_term_edges(p, b_blk);
                    if edges == 0 {
                        continue;
                    }
                    let p_blk = Block(p as u32);
                    if !self.phi_redirect_safe(t, b_blk, p_blk) {
                        all_redirected = false;
                        continue;
                    }
                    self.rewrite_term_target(p, b_blk, t_blk);
                    self.add_redirected_phi_edges(t, b_blk, p_blk, edges);
                    if !preds[t].contains(&p) {
                        preds[t].push(p);
                    }
                    any_redirected = true;
                    changed = true;
                }
                if any_redirected && all_redirected {
                    self.blocks[b].clear();
                    self.remove_all_phi_edges(t, b_blk);
                }
            }

            // Step 2: merge single-predecessor linear blocks (`a -> b`).
            let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
            let mut pred_edges = vec![0u32; self.blocks.len()];
            for &b_u32 in &order {
                for s in self.successors(b_u32 as usize) {
                    pred_edges[s.0 as usize] += 1;
                }
            }
            let mut any_phi_repl = false;
            for &a_u32 in &order {
                let a = a_u32 as usize;
                if self.blocks[a].is_empty() {
                    continue;
                }
                let a_blk = Block(a_u32);
                while let Some(&term_i) = self.blocks[a].last() {
                    let Op::Br(b_blk) = self.insts[term_i as usize].op else { break };
                    let b = b_blk.0 as usize;
                    if b == 0 || b == a || self.blocks[b].is_empty() || pred_edges[b] != 1 {
                        break;
                    }
                    // Verify every leading phi in `b` has an incoming value from `a` before mutating.
                    let mut phis_ok = true;
                    for &inst in &self.blocks[b] {
                        let Op::Phi(inc) = &self.insts[inst as usize].op else { break };
                        if !inc.iter().any(|&(_, bb)| bb == a_blk) {
                            phis_ok = false;
                            break;
                        }
                    }
                    if !phis_ok {
                        break;
                    }
                    // Pop `Br(b)` from `a` and move `b`'s non-phi instructions into `a`.
                    self.blocks[a].pop();
                    let b_insts = std::mem::take(&mut self.blocks[b]);
                    for inst in b_insts {
                        if let Op::Phi(inc) = &self.insts[inst as usize].op {
                            let v = inc.iter().find(|&&(_, bb)| bb == a_blk).unwrap().0;
                            if phi_repl.is_empty() {
                                phi_repl.resize(self.insts.len(), None);
                            }
                            phi_repl[inst as usize] = Some(v);
                            any_phi_repl = true;
                        } else {
                            self.def_block[inst as usize] = a_blk;
                            self.blocks[a].push(inst);
                        }
                    }
                    // Update successor phis from `b` to `a`.
                    let mut seen_s = Vec::new();
                    for s_blk in self.successors(a) {
                        if seen_s.contains(&s_blk) {
                            continue;
                        }
                        seen_s.push(s_blk);
                        let s = s_blk.0 as usize;
                        for k in 0..self.blocks[s].len() {
                            let i = self.blocks[s][k] as usize;
                            let Op::Phi(inc) = &mut self.insts[i].op else { break };
                            for (_, bb) in inc.iter_mut() {
                                if *bb == b_blk {
                                    *bb = a_blk;
                                }
                            }
                        }
                    }
                    pred_edges[b] = 0;
                    changed = true;
                }
            }
            if any_phi_repl {
                let order: Vec<u32> = self.block_layout().0.into_iter().map(|b| b as u32).collect();
                for &b_u32 in &order {
                    let b = b_u32 as usize;
                    for k in 0..self.blocks[b].len() {
                        let i = self.blocks[b][k] as usize;
                        self.insts[i].op.map_operands(|v| resolve(&mut phi_repl, v));
                    }
                }
                phi_repl.fill(None);
            }

            if !changed {
                break;
            }
            ever_changed = true;
        }
        ever_changed
    }

    fn count_term_edges(&self, b: usize, target: Block) -> usize {
        let Some(&term_i) = self.blocks[b].last() else { return 0 };
        match &self.insts[term_i as usize].op {
            Op::Br(t) => usize::from(*t == target),
            Op::CondBr(_, t, f) => usize::from(*t == target) + usize::from(*f == target),
            Op::Switch(_, d, cases) => usize::from(*d == target) + cases.iter().filter(|&&(_, c)| c == target).count(),
            _ => 0,
        }
    }

    fn rewrite_term_target(&mut self, b: usize, from: Block, to: Block) {
        let Some(&term_i) = self.blocks[b].last() else { return };
        match &mut self.insts[term_i as usize].op {
            Op::Br(t) => {
                if *t == from {
                    *t = to;
                }
            }
            Op::CondBr(_, t, f) => {
                if *t == from {
                    *t = to;
                }
                if *f == from {
                    *f = to;
                }
            }
            Op::Switch(_, d, cases) => {
                if *d == from {
                    *d = to;
                }
                for (_, c) in cases.iter_mut() {
                    if *c == from {
                        *c = to;
                    }
                }
            }
            _ => {}
        }
    }

    fn phi_redirect_safe(&self, target: usize, via: Block, pred: Block) -> bool {
        for &i in &self.blocks[target] {
            let Op::Phi(inc) = &self.insts[i as usize].op else { break };
            let Some(&(v_via, _)) = inc.iter().find(|&&(_, bb)| bb == via) else {
                return false;
            };
            if inc.iter().any(|&(v_p, bb)| bb == pred && v_p != v_via) {
                return false;
            }
        }
        true
    }

    fn add_redirected_phi_edges(&mut self, target: usize, via: Block, pred: Block, count: usize) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let v_via = inc.iter().find(|&&(_, bb)| bb == via).unwrap().0;
            for _ in 0..count {
                inc.push((v_via, pred));
            }
        }
    }

    fn remove_all_phi_edges(&mut self, target: usize, from: Block) {
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            inc.retain(|&(_, bb)| bb != from);
        }
    }

    fn drop_phi_edges(&mut self, target: usize, from: Block, count: usize) {
        if count == 0 {
            return;
        }
        for k in 0..self.blocks[target].len() {
            let i = self.blocks[target][k] as usize;
            let Op::Phi(inc) = &mut self.insts[i].op else { break };
            let mut rem = count;
            inc.retain(|&(_, bb)| {
                if rem > 0 && bb == from {
                    rem -= 1;
                    false
                } else {
                    true
                }
            });
        }
    }
}

/// The value `v` stands for after replacements (path-compressing the chain).
fn resolve(repl: &mut [Option<Value>], v: Value) -> Value {
    let mut r = v;
    while let Value::Inst(i) = r {
        match repl[i as usize] {
            Some(n) => r = n,
            None => break,
        }
    }
    let mut c = v;
    while let Value::Inst(i) = c {
        match repl[i as usize] {
            Some(n) if n != r => {
                repl[i as usize] = Some(r);
                c = n;
            }
            _ => break,
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_calls(m: &Module, target_fn: u32) -> usize {
        m.blocks
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&i| matches!(m.insts[i as usize].op, Op::Call(f, _) if f == target_fn))
            .count()
    }

    #[test]
    fn block_cse_invalidates_wave_and_quad_calls_across_discard() {
        let build = |op_name: &str, with_discard: bool| -> (Module, u32) {
            let mut m = Module::new();
            m.define_entry("main");
            let b = m.entry_block();
            m.set_block(b);

            let void = m.void();
            let i1 = m.i1();
            let i32t = m.i32();

            let wave_ty = m.func_ty(i32t, &[i32t, i1]);
            let wave_fn = m.declare(op_name, wave_ty, FnAttrs::NoUnwind);
            let discard_ty = m.func_ty(void, &[i32t, i1]);
            let discard_fn = m.declare("dx.op.discard", discard_ty, FnAttrs::NoUnwind);
            let sink_ty = m.func_ty(void, &[i32t, i32t]);
            let sink_fn = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

            let c_op = m.i32c(116);
            let c_true = m.i1c(true);
            let c_discard_op = m.i32c(82);

            let first = m.call(wave_fn, &[c_op, c_true]);
            if with_discard {
                m.call(discard_fn, &[c_discard_op, c_true]);
            }
            let second = m.call(wave_fn, &[c_op, c_true]);
            m.call(sink_fn, &[first, second]);
            m.ret();
            m.optimize();
            (m, wave_fn)
        };

        // Without `dx.op.discard`, identical wave/quad calls in the same block still CSE to 1.
        let (m_no_discard, f_no_discard) = build("dx.op.waveActiveBallot", false);
        assert_eq!(count_calls(&m_no_discard, f_no_discard), 1);

        // Across `dx.op.discard`, active lanes change so both calls must be preserved.
        for op_name in ["dx.op.waveActiveBallot", "dx.op.quadReadLaneAt"] {
            let (m_with_discard, f_with_discard) = build(op_name, true);
            assert_eq!(
                count_calls(&m_with_discard, f_with_discard),
                2,
                "{op_name} must not be CSE-merged across dx.op.discard"
            );
        }
    }

    fn reachable_blocks(m: &Module) -> usize {
        m.block_layout().0.len()
    }

    #[test]
    fn simplify_cfg_collapses_trampoline_chain_and_merges_linear_blocks() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let t1 = m.new_block();
        let t2 = m.new_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let c7 = m.i32c(7);
        m.br(t1);

        m.set_block(t1);
        m.br(t2);

        m.set_block(t2);
        m.br(mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, c7, t2);
        let c3 = m.i32c(3);
        let sum = m.bin(BinOp::Add, p, c3);
        m.br(exit);

        m.set_block(exit);
        m.call(sink, &[sum]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "entire linear trampoline chain should collapse into entry");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_preserves_phi_conflict_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v1 = m.i32c(1);
        let v2 = m.i32c(2);
        m.cond_br(cond, tramp, join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, entry);
        m.add_incoming(p, v2, tramp);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `tramp` cannot be bypassed because `entry` already feeds a different value into `join`'s phi.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_cfg_collapses_same_target_cond_br_and_switch() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let mid = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let z = m.i32c(0);
        let c1 = m.i32c(1);
        let cond = m.cmp(Pred::Eq, lane, z);
        let v = m.i32c(42);
        m.cond_br(cond, mid, mid);

        m.set_block(mid);
        let p = m.phi(i32t);
        m.add_incoming(p, v, entry);
        m.add_incoming(p, v, entry);
        m.switch(lane, exit, &[(z, exit), (c1, exit)]);

        m.set_block(exit);
        let p2 = m.phi(i32t);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.add_incoming(p2, p, mid);
        m.call(sink, &[p2]);
        m.ret();

        m.optimize();
        assert_eq!(reachable_blocks(&m), 1, "degenerate CondBr and Switch should collapse and merge");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        assert_eq!(count_calls(&m, sink), 1);
    }

    #[test]
    fn simplify_cfg_merges_loop_latch_and_rewrites_header_phi() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let header = m.new_block();
        let latch = m.new_block();
        let exit = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let z = m.i32c(0);
        let one = m.i32c(1);
        let limit = m.i32c(10);
        m.br(header);

        m.set_block(header);
        let iv = m.phi(i32t);
        m.add_incoming(iv, z, entry);
        m.br(latch);

        m.set_block(latch);
        let next = m.bin(BinOp::Add, iv, one);
        m.add_incoming(iv, next, latch);
        let cont = m.cmp(Pred::ULt, next, limit);
        m.cond_br(cont, header, exit);

        m.set_block(exit);
        m.call(sink, &[next]);
        m.ret();

        m.optimize();
        // `latch` merges into `header`, leaving `entry`, `header` (self-loop), `exit`.
        assert_eq!(reachable_blocks(&m), 3);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
        // Header phi's backedge predecessor must now be `header` itself.
        let Value::Inst(iv_i) = iv else { unreachable!() };
        let Op::Phi(inc) = &m.insts[iv_i as usize].op else { panic!("expected phi") };
        assert!(inc.iter().any(|&(_, b)| b == header), "header phi backedge should point to header after latch merge");
        assert!(!inc.iter().any(|&(_, b)| b == latch), "merged latch must not remain in header phi");
    }

    #[test]
    fn simplify_cfg_does_not_inflate_multi_pred_phi_trampoline() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        let left = m.new_block();
        let right = m.new_block();
        let other = m.new_block();
        let tramp = m.new_block();
        let join = m.new_block();

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        m.set_block(entry);
        let lane = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c1 = m.i32c(1);
        let v1 = m.i32c(11);
        let v2 = m.i32c(22);
        m.switch(lane, other, &[(c0, left), (c1, right)]);

        m.set_block(left);
        m.call(sink, &[c0]);
        m.br(tramp);

        m.set_block(right);
        m.call(sink, &[c1]);
        m.br(tramp);

        m.set_block(other);
        m.br(join);

        m.set_block(tramp);
        m.br(join);

        m.set_block(join);
        let p = m.phi(i32t);
        m.add_incoming(p, v1, tramp);
        m.add_incoming(p, v2, other);
        m.call(sink, &[p]);
        m.ret();

        m.optimize();
        // `other` (1 pred -> phi) IS bypassed into `entry -> join`, while `tramp` (2 preds -> phi) stays
        // so `join`'s phi stays at 2 entries instead of growing to 3.
        let Value::Inst(p_i) = p else { unreachable!() };
        let Op::Phi(inc) = &m.insts[p_i as usize].op else { panic!("expected phi") };
        assert_eq!(inc.len(), 2);
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn simplify_op_folds_bitmasks_double_xor_and_all_ones_or() {
        let mut m = Module::new();
        m.define_entry("main");
        let entry = m.entry_block();
        m.set_block(entry);

        let void = m.void();
        let i32t = m.i32();
        let src_ty = m.func_ty(i32t, &[]);
        let src = m.declare("dx.op.waveGetLaneIndex", src_ty, FnAttrs::ReadNone);
        let sink_ty = m.func_ty(void, &[i32t]);
        let sink = m.declare("dx.op.storeOutput.i32", sink_ty, FnAttrs::NoUnwind);

        let x = m.call(src, &[]);
        let c0 = m.i32c(0);
        let c2 = m.i32c(2);
        let c24 = m.i32c(24);
        let all = m.i32c(u32::MAX);
        let mff = m.i32c(0xFF);
        let mhi = m.i32c(0xFFFF_0000);
        let mlo = m.i32c(0x0000_FFFF);
        let align4 = m.i32c(!3u32);
        let low2 = m.i32c(3);
        let k = m.i32c(0xA5A5_5A5A);

        // 1. All-ones `or` absorption + `ashr 0, x -> 0`
        let or_all = m.bin(BinOp::Or, x, all);
        let ashr_z = m.bin(BinOp::AShr, c0, x);
        // 2. Double `xor` cancellation
        let x1 = m.bin(BinOp::Xor, x, k);
        let x2 = m.bin(BinOp::Xor, x1, k);
        // 3. Redundant and disjoint masks on `lshr`
        let shr = m.bin(BinOp::LShr, x, c24);
        let shr_keep = m.bin(BinOp::And, shr, mff);
        let shr_zero = m.bin(BinOp::And, shr, mhi);
        // 4. Redundant and disjoint masks on `shl`
        let shl = m.bin(BinOp::Shl, x, c2);
        let shl_keep = m.bin(BinOp::And, shl, align4);
        let shl_zero = m.bin(BinOp::And, shl, low2);
        // 5. Nested `and` / `or` masks
        let a1 = m.bin(BinOp::And, x, mff);
        let a_keep = m.bin(BinOp::And, a1, mlo);
        let a_zero = m.bin(BinOp::And, a1, mhi);
        let o1 = m.bin(BinOp::Or, x, mlo);
        let o_keep = m.bin(BinOp::Or, o1, mff);
        let o_all = m.bin(BinOp::Or, o1, mhi);
        // 6. Mask on `zext(i1)`
        let cond = m.cmp(Pred::Eq, x, c0);
        let z = m.cast(CastOp::ZExt, cond, i32t);
        let z_keep = m.bin(BinOp::And, z, mff);
        let z_zero = m.bin(BinOp::And, z, mhi);

        for v in [or_all, ashr_z, x2, shr_keep, shr_zero, shl_keep, shl_zero, a_keep, a_zero, o_keep, o_all, z_keep, z_zero] {
            m.call(sink, &[v]);
        }
        m.ret();
        m.optimize();

        let args: Vec<Value> = m.blocks[entry.0 as usize]
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(f, a) if *f == sink => Some(a[0]),
                _ => None,
            })
            .collect();
        assert_eq!(args.len(), 13);
        assert_eq!(m.const_u64(args[0]), Some(u32::MAX as u64), "or(x, ~0) -> ~0");
        assert_eq!(m.const_u64(args[1]), Some(0), "ashr(0, x) -> 0");
        assert_eq!(args[2], x, "xor(xor(x, k), k) -> x");
        assert_eq!(args[3], shr, "and(lshr(x, 24), 0xFF) -> lshr(x, 24)");
        assert_eq!(m.const_u64(args[4]), Some(0), "and(lshr(x, 24), 0xFFFF0000) -> 0");
        assert_eq!(args[5], shl, "and(shl(x, 2), !3) -> shl(x, 2)");
        assert_eq!(m.const_u64(args[6]), Some(0), "and(shl(x, 2), 3) -> 0");
        assert_eq!(args[7], a1, "and(and(x, 0xFF), 0xFFFF) -> and(x, 0xFF)");
        assert_eq!(m.const_u64(args[8]), Some(0), "and(and(x, 0xFF), 0xFFFF0000) -> 0");
        assert_eq!(args[9], o1, "or(or(x, 0xFFFF), 0xFF) -> or(x, 0xFFFF)");
        assert_eq!(m.const_u64(args[10]), Some(u32::MAX as u64), "or(or(x, 0xFFFF), 0xFFFF0000) -> ~0");
        assert_eq!(args[11], z, "and(zext(i1), 0xFF) -> zext(i1)");
        assert_eq!(m.const_u64(args[12]), Some(0), "and(zext(i1), 0xFFFF0000) -> 0");
        let order: Vec<u32> = m.block_layout().0.into_iter().map(|b| b as u32).collect();
        assert!(m.uses_well_formed(&order));
    }

    #[test]
    fn constant_conditional_removes_edge_and_trivial_phi() {
        let mut m = Module::new();
        let yes = m.new_block();
        let no = m.new_block();
        let join = m.new_block();
        let i32 = m.i32();
        let slot = m.alloca(i32, 4);
        let one = m.i32c(1);
        let cond = m.cmp(Pred::Eq, one, one);
        m.cond_br(cond, yes, no);
        m.set_block(yes);
        let live = m.i32c(7);
        m.br(join);
        m.set_block(no);
        let dead = m.i32c(9);
        m.br(join);
        m.set_block(join);
        let phi = m.phi(i32);
        m.add_incoming(phi, live, yes);
        m.add_incoming(phi, dead, no);
        m.store(slot, phi, 4);
        m.ret();

        m.optimize();

        let reachable = m.block_layout().0;
        assert!(reachable.iter().all(|&b| !m.blocks[b].iter().any(|&i| matches!(m.insts[i as usize].op, Op::CondBr(..)))));
        assert!(m.blocks[no.0 as usize].is_empty());
        assert!(!m.blocks[join.0 as usize].iter().any(|&i| matches!(m.insts[i as usize].op, Op::Phi(_))));
    }

    #[test]
    fn constant_switch_keeps_only_the_selected_predecessor() {
        let mut m = Module::new();
        let default = m.new_block();
        let selected = m.new_block();
        let join = m.new_block();
        let one = m.i32c(1);
        let two = m.i32c(2);
        let selector = m.bin(BinOp::Add, one, one);
        m.switch(selector, default, &[(two, selected)]);
        m.set_block(default);
        m.br(join);
        m.set_block(selected);
        m.br(join);
        m.set_block(join);
        m.ret();

        m.optimize();

        let reachable = m.block_layout().0;
        assert!(reachable.iter().all(|&b| !m.blocks[b].iter().any(|&i| matches!(m.insts[i as usize].op, Op::Switch(..)))));
        assert!(m.blocks[default.0 as usize].is_empty());
    }

    #[test]
    fn branch_condition_folds_a_dominated_branch() {
        let mut m = Module::new();
        let yes = m.new_block();
        let no = m.new_block();
        let proven = m.new_block();
        let impossible = m.new_block();
        let i1 = m.i1();
        let condition = m.alloca(i1, 1);
        let condition = m.load(condition, 1);
        m.cond_br(condition, yes, no);
        m.set_block(yes);
        m.cond_br(condition, proven, impossible);
        m.set_block(no);
        m.ret();
        m.set_block(proven);
        m.ret();
        m.set_block(impossible);
        m.ret();

        m.optimize();

        assert!(m.blocks[impossible.0 as usize].is_empty());
        assert!(m.blocks[yes.0 as usize]
            .iter()
            .all(|&i| !matches!(m.insts[i as usize].op, Op::CondBr(..))));
    }

    #[test]
    fn branch_condition_is_dropped_when_paths_reconverge() {
        let mut m = Module::new();
        let yes = m.new_block();
        let no = m.new_block();
        let join = m.new_block();
        let a = m.new_block();
        let b = m.new_block();
        let i1 = m.i1();
        let condition = m.alloca(i1, 1);
        let condition = m.load(condition, 1);
        m.cond_br(condition, yes, no);
        m.set_block(yes);
        m.br(join);
        m.set_block(no);
        m.br(join);
        m.set_block(join);
        m.cond_br(condition, a, b);
        m.set_block(a);
        m.ret();
        m.set_block(b);
        m.ret();

        m.optimize();

        let reachable = m.block_layout().0;
        assert!(reachable.contains(&(a.0 as usize)));
        assert!(reachable.contains(&(b.0 as usize)));
        assert!(reachable.iter().any(|&block| m.blocks[block]
            .iter()
            .any(|&i| matches!(m.insts[i as usize].op, Op::CondBr(..)))));
    }

    #[test]
    fn edge_trim_keeps_a_loop_phi_backedge() {
        let mut m = Module::new();
        let entry = m.entry_block();
        let dead = m.new_block();
        let lp = m.new_block();
        let exit = m.new_block();
        let i32 = m.i32();
        let i1 = m.i1();
        let result = m.alloca(i32, 4);
        let loop_cond = m.alloca(i1, 1);
        let zero = m.i32c(0);
        let one = m.i32c(1);
        let always = m.cmp(Pred::Eq, one, one);
        m.cond_br(always, lp, dead);
        m.set_block(dead);
        let discarded = m.i32c(99);
        m.br(lp);
        m.set_block(lp);
        let phi = m.phi(i32);
        m.add_incoming(phi, zero, entry);
        m.add_incoming(phi, discarded, dead);
        let next = m.bin(BinOp::Add, phi, one);
        m.add_incoming(phi, next, lp);
        let again = m.load(loop_cond, 1);
        m.cond_br(again, lp, exit);
        m.set_block(exit);
        m.store(result, phi, 4);
        m.ret();

        m.optimize();

        assert!(m.blocks[dead.0 as usize].is_empty());
        let incoming = m.phi_incoming(phi);
        assert_eq!(incoming.len(), 2);
        assert!(incoming.iter().any(|p| p.1 == entry));
        assert!(incoming.iter().any(|p| p.1 == lp));
    }

    #[test]
    fn dead_branch_i64_and_heap_handle_do_not_leave_stale_shader_flags() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::shader::{Feature, Shader};
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let live = s.m.new_block();
        let dead = s.m.new_block();
        let one = s.m.i32c(1);
        let cond = s.m.cmp(Pred::Eq, one, one);
        s.m.cond_br(cond, live, dead);
        s.m.set_block(dead);
        let i64 = s.m.i64();
        let z64 = s.m.cast(CastOp::ZExt, one, i64);
        let _ = s.m.bin(BinOp::Add, z64, z64);
        s.require(Feature::Int64Ops);
        let _ = s.handle_from_heap(one, false, false);
        s.m.ret();
        s.m.set_block(live);
        s.m.ret();

        s.m.optimize();
        s.infer_type_features();
        assert!(!s.features.contains(&Feature::Int64Ops), "pruned i64 branch must clear Int64Ops");
        assert!(!s.features.contains(&Feature::ResourceHeapIndexing), "pruned heap handle must clear ResourceHeapIndexing");
    }

    #[test]
    fn local_alloca_store_load_forwarding_and_dse() {
        let mut m = Module::new();
        m.define_entry("main");
        let i32 = m.i32();
        let slot = m.alloca(i32, 4);
        let out = m.global("out", i32, 3, None, false, 4);
        let c1 = m.i32c(11);
        let c2 = m.i32c(22);
        m.store(slot, c1, 4);
        let v1 = m.load(slot, 4);
        m.store(slot, c2, 4);
        let v2 = m.load(slot, 4);
        let sum = m.bin(BinOp::Add, v1, v2);
        m.store(out, sum, 4);
        m.ret();

        m.optimize();

        let ops: Vec<&Op> = m.blocks[0].iter().map(|&i| &m.insts[i as usize].op).collect();
        assert!(!ops.iter().any(|op| matches!(op, Op::Alloca(..))), "scalar alloca must be eliminated: {ops:?}");
        assert!(!ops.iter().any(|op| matches!(op, Op::Load(..))), "forwarded loads must be removed: {ops:?}");
        assert!(
            ops.iter().any(|op| matches!(op, Op::Store(p, v, _) if *p == out && m.const_u64(*v) == Some(33))),
            "store must fold forwarded 11 + 22 == 33: {ops:?}"
        );
    }

    #[test]
    fn hoist_entry_readnone_and_keep_heap_handles_unshared() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::shader::Shader;
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let i32 = s.m.i32();
        let cond_ptr = s.m.global("cond", i32, 3, None, false, 4);
        let out = s.m.global("out", i32, 3, None, false, 4);
        let cv = s.m.load(cond_ptr, 4);
        let zero = s.m.i32c(0);
        let cond = s.m.cmp(Pred::Ne, cv, zero);
        let b1 = s.m.new_block();
        let b2 = s.m.new_block();
        let join = s.m.new_block();
        s.m.cond_br(cond, b1, b2);

        s.m.set_block(b1);
        let g1 = s.group_id(0);
        let h1 = s.heap_raw_buffer_uav(zero, false);
        let r1 = s.raw_load(h1, zero, 1)[0];
        let v1 = s.m.bin(BinOp::Add, g1, r1);
        s.m.br(join);

        s.m.set_block(b2);
        let g2 = s.group_id(0);
        let h2 = s.heap_raw_buffer_uav(zero, false);
        let r2 = s.raw_load(h2, zero, 1)[0];
        let v2 = s.m.bin(BinOp::Add, g2, r2);
        s.m.br(join);

        s.m.set_block(join);
        let p = s.m.phi(i32);
        s.m.add_incoming(p, v1, b1);
        s.m.add_incoming(p, v2, b2);
        s.m.store(out, p, 4);
        s.m.ret();

        s.m.optimize();

        let count_calls = |prefix: &str| -> usize {
            s.m.blocks
                .iter()
                .flatten()
                .filter(|&&i| matches!(s.m.insts[i as usize].op, Op::Call(f, _) if s.m.funcs[f as usize].name.starts_with(prefix)))
                .count()
        };
        assert_eq!(count_calls("dx.op.groupId."), 1, "groupId must hoist to entry block");
        assert_eq!(count_calls("dx.op.createHandleFromHeap"), 2, "createHandleFromHeap must stay unshared across blocks for RDNA4");
        assert_eq!(count_calls("dx.op.annotateHandle"), 2, "annotateHandle must stay block-local for RDNA4");
    }

    #[test]
    fn redundant_barriers_eliminated_within_block() {
        use crate::gpu::dxil::container::Stage;
        use crate::gpu::dxil::ops::barrier;
        use crate::gpu::dxil::shader::Shader;
        let mut s = Shader::new(Stage::Compute, (6, 6));
        s.barrier(barrier::ALL_SYNC);
        s.barrier(barrier::GROUP_SYNC);
        s.barrier(barrier::ALL_SYNC);
        s.m.ret();

        s.m.optimize();

        let barriers = s.m.blocks[0]
            .iter()
            .filter(|&&i| matches!(s.m.insts[i as usize].op, Op::Call(f, _) if s.m.funcs[f as usize].name == "dx.op.barrier"))
            .count();
        assert_eq!(barriers, 1, "adjacent covered barriers must collapse to one");
    }
}
