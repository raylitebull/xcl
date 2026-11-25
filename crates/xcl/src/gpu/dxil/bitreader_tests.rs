//! A test-only reader for the bitstream [`super::bitstream::BitWriter`]
//! produces (both unabbreviated records and `DEFINE_ABBREV` abbreviated records),
//! used to check record layouts that the validator and disassembler do not
//! distinguish (e.g. atomic ordering codes the LLVM reader maps leniently).

use super::module::{BinOp, Module, RmwOp};

#[derive(Debug, Default)]
pub(super) struct Block {
    pub id: u32,
    pub records: Vec<(u32, Vec<u64>)>,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug)]
enum AbOp {
    Literal(u64),
    Fixed(u32),
    Vbr(u32),
    ArrayFixed(u32),
    ArrayVbr(u32),
}

impl Block {
    /// Depth-first search for the first block with `id`.
    pub fn find(&self, id: u32) -> Option<&Block> {
        if self.id == id {
            return Some(self);
        }
        self.blocks.iter().find_map(|b| b.find(id))
    }
    pub fn records_with(&self, code: u32) -> Vec<&[u64]> {
        self.records.iter().filter(|r| r.0 == code).map(|r| &r.1[..]).collect()
    }
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize, // bit position
}

impl Reader<'_> {
    fn fixed(&mut self, w: u32) -> u64 {
        let mut v = 0u64;
        for i in 0..w {
            let bit = (self.b[self.pos / 8] >> (self.pos % 8)) & 1;
            v |= (bit as u64) << i;
            self.pos += 1;
        }
        v
    }
    fn vbr(&mut self, w: u32) -> u64 {
        let hi = 1u64 << (w - 1);
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let c = self.fixed(w);
            v |= (c & (hi - 1)) << shift;
            shift += w - 1;
            if c & hi == 0 {
                return v;
            }
        }
    }
    fn align32(&mut self) {
        self.pos = self.pos.div_ceil(32) * 32;
    }
    fn block(&mut self, id: u32, width: u32, end_bit: Option<usize>) -> Block {
        let mut blk = Block { id, ..Default::default() };
        let mut abbrevs: Vec<Vec<AbOp>> = Vec::new();
        loop {
            if end_bit.is_none() && self.pos + width as usize > self.b.len() * 8 {
                return blk;
            }
            match self.fixed(width) {
                0 => {
                    self.align32();
                    if let Some(e) = end_bit {
                        assert_eq!(self.pos, e, "block length mismatch for block {id}");
                    }
                    return blk;
                }
                1 => {
                    let sub = self.vbr(8) as u32;
                    let w = self.vbr(4) as u32;
                    self.align32();
                    let words = self.fixed(32) as usize;
                    let end = self.pos + words * 32;
                    blk.blocks.push(self.block(sub, w, Some(end)));
                }
                2 => {
                    let n = self.vbr(5);
                    let mut ops = Vec::new();
                    let mut i = 0;
                    while i < n {
                        if self.fixed(1) == 1 {
                            ops.push(AbOp::Literal(self.vbr(8)));
                            i += 1;
                        } else {
                            match self.fixed(3) {
                                1 => {
                                    ops.push(AbOp::Fixed(self.vbr(5) as u32));
                                    i += 1;
                                }
                                2 => {
                                    ops.push(AbOp::Vbr(self.vbr(5) as u32));
                                    i += 1;
                                }
                                3 => {
                                    assert_eq!(self.fixed(1), 0, "array element must be encoded");
                                    match self.fixed(3) {
                                        1 => ops.push(AbOp::ArrayFixed(self.vbr(5) as u32)),
                                        2 => ops.push(AbOp::ArrayVbr(self.vbr(5) as u32)),
                                        e => panic!("unsupported array elem encoding {e}"),
                                    }
                                    i += 2;
                                }
                                e => panic!("unsupported abbrev encoding {e}"),
                            }
                        }
                    }
                    abbrevs.push(ops);
                }
                3 => {
                    let code = self.vbr(6) as u32;
                    let n = self.vbr(6);
                    let ops = (0..n).map(|_| self.vbr(6)).collect();
                    blk.records.push((code, ops));
                }
                a => {
                    let ab = abbrevs.get((a - 4) as usize).unwrap_or_else(|| panic!("unknown abbreviation id {a}"));
                    let mut vals = Vec::new();
                    for op in ab {
                        match *op {
                            AbOp::Literal(v) => vals.push(v),
                            AbOp::Fixed(w) => vals.push(self.fixed(w)),
                            AbOp::Vbr(w) => vals.push(self.vbr(w)),
                            AbOp::ArrayFixed(w) => {
                                let len = self.vbr(6);
                                for _ in 0..len {
                                    vals.push(self.fixed(w));
                                }
                            }
                            AbOp::ArrayVbr(w) => {
                                let len = self.vbr(6);
                                for _ in 0..len {
                                    vals.push(self.vbr(w));
                                }
                            }
                        }
                    }
                    let code = vals[0] as u32;
                    blk.records.push((code, vals[1..].to_vec()));
                }
            }
        }
    }
}

/// Parses bitcode (with the `BC\xC0\xDE` magic) into its block tree; the
/// root is a pseudo block with id `u32::MAX`.
pub(super) fn parse(bc: &[u8]) -> Block {
    assert_eq!(&bc[..4], b"BC\xC0\xDE");
    let mut r = Reader { b: &bc[4..], pos: 0 };
    r.block(u32::MAX, 2, None)
}

const BLOCK_FUNCTION: u32 = 12;
const FUNC_CODE_INST_BINOP: u32 = 2;
const FUNC_CODE_INST_ATOMICRMW: u32 = 38;
const FUNC_CODE_INST_CMPXCHG: u32 = 46;
/// `bitc::ORDERING_SEQCST` (the in-memory `AtomicOrdering` value is 7, the
/// bitcode code is 6).
const ORDERING_SEQCST: u64 = 6;

fn atomic_module() -> Vec<u8> {
    let mut m = Module::new();
    m.define_entry("main");
    let i32 = m.i32();
    let arr = m.array(4, i32);
    let g = m.global("g", arr, 3, None, false, 4);
    let zero = m.i32c(0);
    let p = m.gep(g, &[zero, zero]);
    let one = m.i32c(1);
    let a = m.atomic_rmw(RmwOp::Add, p, one);
    let b = m.cmpxchg(p, a, one);
    let _ = m.bin(BinOp::Add, a, b);
    m.ret();
    m.to_bitcode("dxil-ms-dx", "")
}

#[test]
fn reader_round_trips_block_structure() {
    let bc = atomic_module();
    let root = parse(&bc);
    assert!(root.find(8).is_some(), "module block");
    assert!(root.find(BLOCK_FUNCTION).is_some(), "function block");
}

#[test]
fn a_shorter_binop_record_has_only_its_own_operands() {
    let mut m = Module::new();
    m.define_entry("main");
    let one = m.i32c(1);
    let sum = m.bin_flags(BinOp::Add, one, one, 3);
    m.bin(BinOp::Xor, sum, one);
    m.ret();

    let root = parse(&m.to_bitcode("dxil-ms-dx", ""));
    let records = root.find(BLOCK_FUNCTION).unwrap().records_with(FUNC_CODE_INST_BINOP);
    assert_eq!(records, [&[1, 1, BinOp::Add as u64, 3][..], &[1, 2, BinOp::Xor as u64][..]]);
}

#[test]
fn atomics_use_bitcode_ordering_codes() {
    let root = parse(&atomic_module());
    let f = root.find(BLOCK_FUNCTION).unwrap();
    let rmw = f.records_with(FUNC_CODE_INST_ATOMICRMW);
    assert_eq!(rmw.len(), 1);
    // [ptr, val, op, vol, ordering, synchscope]
    assert_eq!(rmw[0][4], ORDERING_SEQCST, "atomicrmw ordering: {:?}", rmw[0]);
    assert_eq!(rmw[0][5], 1, "crossthread synch scope");
    let cx = f.records_with(FUNC_CODE_INST_CMPXCHG);
    assert_eq!(cx.len(), 1);
    // [ptr, cmp, new, vol, success, synchscope, failure, weak]
    assert_eq!(cx[0][4], ORDERING_SEQCST, "cmpxchg success ordering: {:?}", cx[0]);
    assert_eq!(cx[0][6], ORDERING_SEQCST, "cmpxchg failure ordering: {:?}", cx[0]);
}

#[test]
fn float_constants_ignore_bits_above_their_width() {
    let mut m = Module::new();
    let (f16, f32) = (m.f16(), m.f32());
    let one = m.f32c(1.0f32.to_bits());
    assert_eq!(m.const_float(f32, 0xFFFF_0000_0000_0000 | 1.0f32.to_bits() as u64), one);
    let h = m.const_float(f16, 0x3C00);
    assert_eq!(m.const_float(f16, 0x1_3C00), h);
}
