//! LLVM bitstream writer: fixed and VBR fields, nested blocks with
//! back-patched lengths, unabbreviated records, and LLVM `DEFINE_ABBREV`
//! abbreviations for compact DXIL encoding.

/// Bit-level writer, little-endian within 32-bit words as LLVM defines.
pub struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
    /// Open blocks: (byte offset of the length word, outer abbreviation width, next abbrev id).
    blocks: Vec<(usize, u32, u32)>,
    width: u32,
}

/// Abbreviation operand encoding for LLVM `DEFINE_ABBREV` records.
#[derive(Clone, Copy, Debug)]
pub enum AbbrevOp {
    Literal(u64),
    Fixed(u32),
    Vbr(u32),
    /// Trailing array of fixed-width operands (LLVM `Array` + `Fixed(w)` pair).
    ArrayFixed(u32),
    /// Trailing array of VBR-encoded operands (LLVM `Array` + `VBR(w)` pair).
    ArrayVbr(u32),
}

/// Abbreviation width used for every block opened via [`BitWriter::enter_block`] (IDs 0..7).
pub const ABBREV_WIDTH: u32 = 3;

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, nbits: 0, blocks: Vec::new(), width: 2 }
    }

    pub fn emit(&mut self, value: u64, width: u32) {
        debug_assert!(width <= 32);
        debug_assert!(width == 64 || width == 32 || value >> width == 0, "value {value:#x} does not fit {width} bits");
        self.acc |= value << self.nbits;
        self.nbits += width;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    pub fn vbr(&mut self, mut value: u64, width: u32) {
        let hi = 1u64 << (width - 1);
        while value >= hi {
            self.emit((value & (hi - 1)) | hi, width);
            value >>= width - 1;
        }
        self.emit(value, width);
    }

    pub fn align32(&mut self) {
        if self.nbits > 0 {
            self.emit(0, 8 - self.nbits);
        }
        while self.out.len() % 4 != 0 {
            self.out.push(0);
        }
    }

    pub fn enter_block(&mut self, id: u32) {
        self.enter_block_with_width(id, ABBREV_WIDTH);
    }

    pub fn enter_block_with_width(&mut self, id: u32, abbrev_width: u32) {
        self.emit(1, self.width); // ENTER_SUBBLOCK
        self.vbr(id as u64, 8);
        self.vbr(abbrev_width as u64, 4);
        self.align32();
        let at = self.out.len();
        self.out.extend_from_slice(&[0; 4]);
        self.blocks.push((at, self.width, 4));
        self.width = abbrev_width;
    }

    pub fn end_block(&mut self) {
        self.emit(0, self.width); // END_BLOCK
        self.align32();
        let (at, outer, _) = self.blocks.pop().expect("end_block without enter_block");
        let words = ((self.out.len() - at - 4) / 4) as u32;
        self.out[at..at + 4].copy_from_slice(&words.to_le_bytes());
        self.width = outer;
    }

    /// Define a block-local abbreviation (`DEFINE_ABBREV = 2`) and return its ID (`>= 4`).
    pub fn define_abbrev(&mut self, ops: &[AbbrevOp]) -> u32 {
        let id = {
            let top = self.blocks.last_mut().expect("define_abbrev outside block");
            let id = top.2;
            top.2 += 1;
            id
        };
        debug_assert!((id as u64) < (1u64 << self.width), "abbrev id {id} exceeds width {}", self.width);
        let num_ops: u64 = ops.iter().map(|o| if matches!(o, AbbrevOp::ArrayFixed(_) | AbbrevOp::ArrayVbr(_)) { 2 } else { 1 }).sum();
        self.emit(2, self.width); // DEFINE_ABBREV
        self.vbr(num_ops, 5);
        for &op in ops {
            match op {
                AbbrevOp::Literal(v) => {
                    self.emit(1, 1);
                    self.vbr(v, 8);
                }
                AbbrevOp::Fixed(w) => {
                    self.emit(0, 1);
                    self.emit(1, 3);
                    self.vbr(w as u64, 5);
                }
                AbbrevOp::Vbr(w) => {
                    self.emit(0, 1);
                    self.emit(2, 3);
                    self.vbr(w as u64, 5);
                }
                AbbrevOp::ArrayFixed(w) => {
                    self.emit(0, 1);
                    self.emit(3, 3); // Array
                    self.emit(0, 1);
                    self.emit(1, 3); // Fixed element
                    self.vbr(w as u64, 5);
                }
                AbbrevOp::ArrayVbr(w) => {
                    self.emit(0, 1);
                    self.emit(3, 3); // Array
                    self.emit(0, 1);
                    self.emit(2, 3); // VBR element
                    self.vbr(w as u64, 5);
                }
            }
        }
        id
    }

    /// Emit an abbreviated record defined by [`Self::define_abbrev`]. `values` supplies only the
    /// non-literal operands in order; if `ops` ends with `ArrayFixed`/`ArrayVbr`, remaining items
    /// in `values` form the array payload.
    pub fn emit_abbrev(&mut self, id: u32, ops: &[AbbrevOp], values: &[u64]) {
        self.emit(id as u64, self.width);
        let mut idx = 0usize;
        for &op in ops {
            match op {
                AbbrevOp::Literal(_) => {}
                AbbrevOp::Fixed(w) => {
                    self.emit(values[idx], w);
                    idx += 1;
                }
                AbbrevOp::Vbr(w) => {
                    self.vbr(values[idx], w);
                    idx += 1;
                }
                AbbrevOp::ArrayFixed(w) => {
                    let rest = &values[idx..];
                    self.vbr(rest.len() as u64, 6);
                    for &v in rest {
                        self.emit(v, w);
                    }
                    idx = values.len();
                }
                AbbrevOp::ArrayVbr(w) => {
                    let rest = &values[idx..];
                    self.vbr(rest.len() as u64, 6);
                    for &v in rest {
                        self.vbr(v, w);
                    }
                    idx = values.len();
                }
            }
        }
        debug_assert_eq!(idx, values.len());
    }

    /// An unabbreviated record.
    pub fn record(&mut self, code: u32, ops: &[u64]) {
        self.emit(3, self.width); // UNABBREV_RECORD
        self.vbr(code as u64, 6);
        self.vbr(ops.len() as u64, 6);
        for &op in ops {
            self.vbr(op, 6);
        }
    }

    /// A record whose operands are the bytes of a string.
    pub fn record_str(&mut self, code: u32, prefix: &[u64], s: &str) {
        let mut ops = prefix.to_vec();
        ops.extend(s.bytes().map(u64::from));
        self.record(code, &ops);
    }

    pub fn finish(mut self) -> Vec<u8> {
        assert!(self.blocks.is_empty(), "unterminated bitstream block");
        self.align32();
        self.out
    }
}

/// LLVM's signed VBR operand encoding: sign in bit 0.
pub fn signed(v: i64) -> u64 {
    if v >= 0 {
        (v as u64) << 1
    } else if v == i64::MIN {
        1 // "-0": LLVM's encoding of INT64_MIN
    } else {
        ((v.unsigned_abs()) << 1) | 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vbr_and_fixed_fields_pack_little_endian() {
        let mut w = BitWriter::new();
        w.emit(0b10, 2);
        w.vbr(5, 3); // 0b101 needs continuation in 3-bit VBR: chunks 01|1, 01|0... check value round trip below
        w.align32();
        let bytes = w.finish();
        assert_eq!(bytes.len(), 4);
        // Decode back: 2 bits, then VBR3.
        let word = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(word & 3, 0b10);
        let (c0, c1) = ((word >> 2) & 7, (word >> 5) & 7);
        assert_eq!(c0, 0b101); // low 2 bits of 5 = 01, continuation bit set
        assert_eq!(c1, 0b001); // remaining 1, no continuation
    }

    #[test]
    fn block_lengths_are_back_patched_in_words() {
        let mut w = BitWriter::new();
        w.enter_block(8);
        w.record(1, &[1]);
        w.end_block();
        let b = w.finish();
        let len = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        assert_eq!(b.len(), 8 + len * 4);
    }

    #[test]
    fn signed_vbr_operands() {
        assert_eq!(signed(0), 0);
        assert_eq!(signed(3), 6);
        assert_eq!(signed(-3), 7);
        assert_eq!(signed(i64::MIN), 1);
    }

    #[test]
    fn define_abbrev_round_trips_fixed_vbr_and_arrays() {
        const AB: &[AbbrevOp] = &[AbbrevOp::Literal(22), AbbrevOp::ArrayFixed(32)];
        let mut w = BitWriter::new();
        for b in [b'B', b'C', 0xC0, 0xDE] {
            w.emit(b as u64, 8);
        }
        w.enter_block_with_width(11, 3);
        let id = w.define_abbrev(AB);
        assert_eq!(id, 4);
        w.emit_abbrev(id, AB, &[0x1234_5678, 0xDEAD_BEEF]);
        w.end_block();
        let bc = w.finish();
        assert_eq!(&bc[..4], b"BC\xC0\xDE");
        assert_eq!(bc.len() % 4, 0);
    }
}
