//! The graphics pipelines a title used, as it described them, kept for the shader preparation of
//! its next run (`shaders::prepare_for_launch`): it builds them before the title starts, the way
//! the title does (`real::prepare_graphics`), so the title finds their translations and the
//! driver's blobs on disk. `used-pipelines.txt` holds only the keys of the shader sets and the
//! driver cache only blobs named after the translated DXIL: neither is enough to make one again.
//!
//! `used-graphics-pipelines.bin` beside the DXIL cache:
//!
//! ```text
//! file     = "XCLGFXP\0", version u32, entry*
//! entry    = length u32, check u32 (the low half of fnv64 of the payload), payload
//! payload  = 1 key u64 shaders [u64; 5] state input*      a pipeline
//!          | 2 key u64 bytes                              a shader (preparation job files only)
//! key      = `real::shaders_key` of the set (as `used-pipelines.txt` has it); in a shader entry
//!            `real::shader_key` of it
//! shaders  = `real::shader_key` of the VS, PS, DS, HS, GS (0: none)
//! state    = every field of the description but its pointers, in `state`'s order (varints)
//! input    = count, then each element's semantic name (length, bytes), index, format, slot,
//!            offset, class, step rate (varints)
//! ```
//!
//! Integers little-endian, varints LEB128. A pipeline entry holds no shader code: the preparation
//! takes the shaders from the title's shader files by key, as it does those of `used-shaders.txt`
//! (~200 bytes a pipeline). A file of another version is read as none and started again; an entry
//! cut short or damaged ends the file there (the writer cuts it off before it appends). The title
//! side: a pipeline's description is noted at its first use in a run (`note`, from
//! `pso_async::on_first_use`), each once, written by a thread of its own, the file at most
//! `XCL_GFX_RECORDS_MB` (64).

use std::collections::HashSet;
use std::ffi::CString;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use windows::Win32::Graphics::Direct3D12::*;

const MAGIC: &[u8; 8] = b"XCLGFXP\0";
/// The layout of an entry: a change makes every file of the older one read as none.
const VERSION: u32 = 1;
const HEADER: usize = 12;
const PIPELINE: u8 = 1;
const SHADER: u8 = 2;

/// The records of this cache (beside the DXIL cache, as `used-pipelines.txt`).
pub(crate) fn path() -> PathBuf {
    crate::gpu::recomp::dxc::cache_dir().join("used-graphics-pipelines.bin")
}

/// `XCL_GFX_RECORDS_MB`: the most the records may take on disk (64 MB: ~300 000 pipelines).
fn cap_bytes() -> u64 {
    cap_of(std::env::var("XCL_GFX_RECORDS_MB").ok().as_deref())
}

fn cap_of(v: Option<&str>) -> u64 {
    v.and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64) << 20
}

fn put_var(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn var(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let (&b, rest) = self.0.split_first()?;
            self.0 = rest;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
        }
        None
    }
    fn u32var(&mut self) -> Option<u32> {
        self.var().and_then(|v| u32::try_from(v).ok())
    }
    fn u64(&mut self) -> Option<u64> {
        let (b, rest) = self.0.split_at_checked(8)?;
        self.0 = rest;
        Some(u64::from_le_bytes(b.try_into().ok()?))
    }
    fn bytes(&mut self, n: usize) -> Option<&[u8]> {
        let (b, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(b)
    }
}

/// Every field of `d` but its pointers (shaders, root signature, input layout, stream output,
/// cached blob), each as a u32 handed to `f`, which may change it: one order for writing and
/// reading. What the title's translated pipeline is made of beside its shaders
/// (`real::graphics_blob_key` hashes the same).
fn state(d: &mut D3D12_GRAPHICS_PIPELINE_STATE_DESC, f: &mut dyn FnMut(&mut u32)) {
    macro_rules! int {
        ($x:expr) => {{
            let mut v = $x as u32;
            f(&mut v);
            $x = v as i32;
        }};
    }
    macro_rules! byte {
        ($x:expr) => {{
            let mut v = $x as u32;
            f(&mut v);
            $x = v as u8;
        }};
    }
    macro_rules! float {
        ($x:expr) => {{
            let mut v = $x.to_bits();
            f(&mut v);
            $x = f32::from_bits(v);
        }};
    }
    let b = &mut d.BlendState;
    int!(b.AlphaToCoverageEnable.0);
    int!(b.IndependentBlendEnable.0);
    for rt in &mut b.RenderTarget {
        int!(rt.BlendEnable.0);
        int!(rt.LogicOpEnable.0);
        int!(rt.SrcBlend.0);
        int!(rt.DestBlend.0);
        int!(rt.BlendOp.0);
        int!(rt.SrcBlendAlpha.0);
        int!(rt.DestBlendAlpha.0);
        int!(rt.BlendOpAlpha.0);
        int!(rt.LogicOp.0);
        byte!(rt.RenderTargetWriteMask);
    }
    f(&mut d.SampleMask);
    let r = &mut d.RasterizerState;
    int!(r.FillMode.0);
    int!(r.CullMode.0);
    int!(r.FrontCounterClockwise.0);
    int!(r.DepthBias);
    float!(r.DepthBiasClamp);
    float!(r.SlopeScaledDepthBias);
    int!(r.DepthClipEnable.0);
    int!(r.MultisampleEnable.0);
    int!(r.AntialiasedLineEnable.0);
    f(&mut r.ForcedSampleCount);
    int!(r.ConservativeRaster.0);
    let z = &mut d.DepthStencilState;
    int!(z.DepthEnable.0);
    int!(z.DepthWriteMask.0);
    int!(z.DepthFunc.0);
    int!(z.StencilEnable.0);
    byte!(z.StencilReadMask);
    byte!(z.StencilWriteMask);
    for face in [&mut z.FrontFace, &mut z.BackFace] {
        int!(face.StencilFailOp.0);
        int!(face.StencilDepthFailOp.0);
        int!(face.StencilPassOp.0);
        int!(face.StencilFunc.0);
    }
    int!(d.IBStripCutValue.0);
    int!(d.PrimitiveTopologyType.0);
    f(&mut d.NumRenderTargets);
    for format in &mut d.RTVFormats {
        int!(format.0);
    }
    int!(d.DSVFormat.0);
    f(&mut d.SampleDesc.Count);
    f(&mut d.SampleDesc.Quality);
    f(&mut d.NodeMask);
    int!(d.Flags.0);
}

/// The pipeline entry of `d` (a title's description, its input layout readable), of the shader
/// set `key` whose shaders are `shaders` (`Record::shaders`).
pub(crate) unsafe fn encode(key: u64, shaders: [u64; 5], d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.push(PIPELINE);
    out.extend_from_slice(&key.to_le_bytes());
    for s in shaders {
        out.extend_from_slice(&s.to_le_bytes());
    }
    // (a copy for `state`, which takes the fields by `&mut`; its pointers are not used)
    let mut copy: D3D12_GRAPHICS_PIPELINE_STATE_DESC = std::ptr::read(d);
    state(&mut copy, &mut |v| put_var(&mut out, *v as u64));
    let il = &d.InputLayout;
    let elems = if il.NumElements > 0 && !il.pInputElementDescs.is_null() { std::slice::from_raw_parts(il.pInputElementDescs, il.NumElements as usize) } else { &[] };
    put_var(&mut out, elems.len() as u64);
    for e in elems {
        let name = if e.SemanticName.0.is_null() { &[][..] } else { std::ffi::CStr::from_ptr(e.SemanticName.0 as *const std::ffi::c_char).to_bytes() };
        put_var(&mut out, name.len() as u64);
        out.extend_from_slice(name);
        for v in [e.SemanticIndex, e.Format.0 as u32, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0 as u32, e.InstanceDataStepRate] {
            put_var(&mut out, v as u64);
        }
    }
    out
}

/// A shader entry (the preparation's job files): `bytes`, of `real::shader_key` `key`.
pub(crate) fn shader_entry(key: u64, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(9 + bytes.len());
    out.push(SHADER);
    out.extend_from_slice(&key.to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

/// A pipeline as a title described it, but its shaders and root signature (`desc` puts them in).
pub(crate) struct Record {
    pub key: u64,
    /// `real::shader_key` of the VS, PS, DS, HS, GS (0: none)
    pub shaders: [u64; 5],
    desc: Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC>,
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    _names: Vec<CString>,
}

// (only read once made; the names the inputs point at live as long)
unsafe impl Send for Record {}
unsafe impl Sync for Record {}

impl Record {
    /// The description with the shaders `code` (VS, PS, DS, HS, GS), pointing into `self` and
    /// `code`, with no root signature (a translated pipeline gets XCL's).
    pub(crate) fn desc(&self, code: [Option<&[u8]>; 5]) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // (no root signature in it: nothing is released twice)
        let mut d: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::ptr::read(&*self.desc) };
        let bytecode = |c: Option<&[u8]>| c.map_or_else(D3D12_SHADER_BYTECODE::default, |c| D3D12_SHADER_BYTECODE { pShaderBytecode: c.as_ptr().cast(), BytecodeLength: c.len() });
        let [vs, ps, ds, hs, gs] = code;
        (d.VS, d.PS, d.DS, d.HS, d.GS) = (bytecode(vs), bytecode(ps), bytecode(ds), bytecode(hs), bytecode(gs));
        if !self.inputs.is_empty() {
            d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: self.inputs.as_ptr(), NumElements: self.inputs.len() as u32 };
        }
        d
    }
}

pub(crate) enum Entry {
    Pipeline(Record),
    Shader(u64, Vec<u8>),
}

/// The entry of `payload` (None: not one this version writes).
pub(crate) fn decode(payload: &[u8]) -> Option<Entry> {
    let (&tag, rest) = payload.split_first()?;
    let mut r = Reader(rest);
    let key = r.u64()?;
    if tag == SHADER {
        return Some(Entry::Shader(key, r.0.to_vec()));
    }
    if tag != PIPELINE {
        return None;
    }
    let mut shaders = [0u64; 5];
    for s in &mut shaders {
        *s = r.u64()?;
    }
    let mut desc: Box<D3D12_GRAPHICS_PIPELINE_STATE_DESC> = Box::new(unsafe { std::mem::zeroed() });
    let mut short = false;
    state(&mut desc, &mut |v| match r.u32var() {
        Some(x) => *v = x,
        None => short = true,
    });
    if short {
        return None;
    }
    let n = r.var()? as usize;
    let mut names = Vec::new();
    let mut fields = Vec::new();
    for _ in 0..n.min(64) {
        let len = r.var()? as usize;
        names.push(CString::new(r.bytes(len)?).ok()?);
        let mut f = [0u32; 6];
        for v in &mut f {
            *v = r.u32var()?;
        }
        fields.push(f);
    }
    if n > 64 || !r.0.is_empty() {
        return None;
    }
    let inputs = names
        .iter()
        .zip(&fields)
        .map(|(name, f)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::PCSTR(name.as_ptr().cast()),
            SemanticIndex: f[0],
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(f[1] as i32),
            InputSlot: f[2],
            AlignedByteOffset: f[3],
            InputSlotClass: D3D12_INPUT_CLASSIFICATION(f[4] as i32),
            InstanceDataStepRate: f[5],
        })
        .collect();
    Some(Entry::Pipeline(Record { key, shaders, desc, inputs, _names: names }))
}

fn header() -> [u8; HEADER] {
    let mut h = [0u8; HEADER];
    h[..8].copy_from_slice(MAGIC);
    h[8..].copy_from_slice(&VERSION.to_le_bytes());
    h
}

fn check(payload: &[u8]) -> u32 {
    crate::gpu::shader::fnv64(payload) as u32
}

/// The payloads of a file's whole entries and where the last of them ends. None: no file of this
/// version (another one, or not one at all).
fn payloads(file: &[u8]) -> Option<(Vec<&[u8]>, usize)> {
    if file.len() < HEADER || file[..HEADER] != header() {
        return None;
    }
    let (mut list, mut at) = (Vec::new(), HEADER);
    while let Some(head) = file.get(at..at + 8) {
        let len = u32::from_le_bytes(head[..4].try_into().unwrap()) as usize;
        let Some(payload) = file.get(at + 8..at + 8 + len) else { break };
        if check(payload) != u32::from_le_bytes(head[4..].try_into().unwrap()) {
            break;
        }
        list.push(payload);
        at += 8 + len;
    }
    Some((list, at))
}

/// The entries of the file at `path` (none: no file, or one of another version).
pub(crate) fn read(path: &Path) -> Vec<Vec<u8>> {
    let file = std::fs::read(path).unwrap_or_default();
    payloads(&file).map_or_else(Vec::new, |(list, _)| list.into_iter().map(<[u8]>::to_vec).collect())
}

/// What became of an entry handed to `Writer::add`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Added {
    New,
    /// in the file already
    Known,
    /// not written: the file is at its cap
    Full,
}

/// Appends entries to a file of records, each once, up to `cap` bytes.
pub(crate) struct Writer {
    file: std::fs::File,
    len: u64,
    cap: u64,
    seen: HashSet<u64>,
    full: bool,
}

impl Writer {
    /// The file at `path` for appending: its whole entries are kept (a cut-short tail is cut off),
    /// one of another version is started again.
    pub(crate) fn open(path: &Path, cap: u64) -> std::io::Result<Writer> {
        let old = std::fs::read(path).unwrap_or_default();
        let (seen, end) = match payloads(&old) {
            Some((list, end)) => (list.iter().map(|p| crate::gpu::shader::fnv64(p)).collect(), end),
            None => (HashSet::new(), 0),
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut file = std::fs::OpenOptions::new().write(true).create(true).truncate(false).open(path)?;
        let len = if end == 0 {
            file.set_len(0)?;
            file.write_all(&header())?;
            HEADER as u64
        } else {
            file.set_len(end as u64)?;
            end as u64
        };
        file.seek(std::io::SeekFrom::Start(len))?;
        Ok(Writer { file, len, cap, seen, full: false })
    }

    /// Entries in the file.
    pub(crate) fn count(&self) -> usize {
        self.seen.len()
    }

    pub(crate) fn add(&mut self, payload: &[u8]) -> std::io::Result<Added> {
        let id = crate::gpu::shader::fnv64(payload);
        if self.seen.contains(&id) {
            return Ok(Added::Known);
        }
        let n = 8 + payload.len() as u64;
        if self.full || self.len + n > self.cap {
            self.full = true;
            return Ok(Added::Full);
        }
        let mut entry = Vec::with_capacity(n as usize);
        entry.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        entry.extend_from_slice(&check(payload).to_le_bytes());
        entry.extend_from_slice(payload);
        self.file.write_all(&entry)?;
        self.len += n;
        self.seen.insert(id);
        Ok(Added::New)
    }
}

/// A graphics pipeline's first use in this run: its entry (`encode`) goes to the file on a thread
/// of its own, not the title's.
pub(crate) fn note(payload: Vec<u8>) {
    use std::sync::{mpsc, Mutex, OnceLock};
    // (a test does not write the user's file)
    if cfg!(test) {
        return;
    }
    static TX: OnceLock<Mutex<mpsc::Sender<Vec<u8>>>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        let _ = std::thread::Builder::new().name("xcl-gfx-records".into()).spawn(move || {
            let (path, cap) = (path(), cap_bytes());
            let mut w = match Writer::open(&path, cap) {
                Ok(w) => w,
                Err(e) => {
                    crate::xlog_warn!("D3D12.X: {}: {e} – the graphics pipelines of this run are not prepared before the next", path.display());
                    return;
                }
            };
            crate::xlog_info!("D3D12.X: graphics pipelines are recorded at their first use for the next run's preparation ({} from earlier runs)", w.count());
            for payload in rx.iter() {
                match w.add(&payload) {
                    Ok(Added::Full) => {
                        crate::xlog_warn!("D3D12.X: the graphics pipeline records are at their cap of {} MB (XCL_GFX_RECORDS_MB): pipelines first used from here on are not prepared before the next run", cap >> 20);
                        // (the rest are dropped unread)
                        for _ in rx.iter() {}
                        return;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        crate::xlog_warn!("D3D12.X: {}: {e} – no more graphics pipelines are recorded in this run", path.display());
                        return;
                    }
                }
            }
        });
        Mutex::new(tx)
    });
    let _ = tx.lock().unwrap().send(payload);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

    struct Directory(PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn directory(name: &str) -> Directory {
        let dir = std::env::temp_dir().join(format!("xcl-gfx-records-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Directory(dir)
    }

    /// A description with every field set to a value of its own (none its default), written out
    /// field by field here: a field `state` leaves out does not come back.
    fn described() -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        let mut n = 0x10;
        let mut next = || {
            n += 3;
            n
        };
        let b = &mut d.BlendState;
        b.AlphaToCoverageEnable.0 = 1;
        b.IndependentBlendEnable.0 = 1;
        for rt in &mut b.RenderTarget {
            rt.BlendEnable.0 = 1;
            rt.LogicOpEnable.0 = 1;
            rt.SrcBlend.0 = next();
            rt.DestBlend.0 = next();
            rt.BlendOp.0 = next();
            rt.SrcBlendAlpha.0 = next();
            rt.DestBlendAlpha.0 = next();
            rt.BlendOpAlpha.0 = next();
            rt.LogicOp.0 = next();
            rt.RenderTargetWriteMask = next() as u8;
        }
        d.SampleMask = 0xfedc_ba98;
        let r = &mut d.RasterizerState;
        r.FillMode.0 = next();
        r.CullMode.0 = next();
        r.FrontCounterClockwise.0 = 1;
        r.DepthBias = -1234;
        r.DepthBiasClamp = -0.25;
        r.SlopeScaledDepthBias = 3.5;
        r.DepthClipEnable.0 = 1;
        r.MultisampleEnable.0 = 1;
        r.AntialiasedLineEnable.0 = 1;
        r.ForcedSampleCount = 8;
        r.ConservativeRaster.0 = 1;
        let z = &mut d.DepthStencilState;
        z.DepthEnable.0 = 1;
        z.DepthWriteMask.0 = 1;
        z.DepthFunc.0 = next();
        z.StencilEnable.0 = 1;
        z.StencilReadMask = 0x3c;
        z.StencilWriteMask = 0xc3;
        for face in [&mut z.FrontFace, &mut z.BackFace] {
            face.StencilFailOp.0 = next();
            face.StencilDepthFailOp.0 = next();
            face.StencilPassOp.0 = next();
            face.StencilFunc.0 = next();
        }
        d.IBStripCutValue.0 = 2;
        d.PrimitiveTopologyType.0 = 3;
        d.NumRenderTargets = 7;
        for f in &mut d.RTVFormats {
            *f = DXGI_FORMAT(next());
        }
        d.DSVFormat = DXGI_FORMAT(next());
        d.SampleDesc.Count = 4;
        d.SampleDesc.Quality = 9;
        d.NodeMask = 0x5;
        d.Flags.0 = 1;
        d
    }

    fn pipeline(payload: &[u8]) -> Record {
        match decode(payload) {
            Some(Entry::Pipeline(r)) => r,
            _ => panic!("not a pipeline entry"),
        }
    }

    /// A description comes back with every field (as the key of its translated pipeline's driver
    /// blob sees them), its input layout and its shaders' keys.
    #[test]
    fn a_description_comes_back_as_it_was_written() {
        let names = [CString::new("POSITION").unwrap(), CString::new("TEXCOORD").unwrap()];
        let elems = [
            D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(names[0].as_ptr().cast()), SemanticIndex: 0, Format: DXGI_FORMAT(6), InputSlot: 0, AlignedByteOffset: 0, InputSlotClass: D3D12_INPUT_CLASSIFICATION(0), InstanceDataStepRate: 0 },
            D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(names[1].as_ptr().cast()), SemanticIndex: 3, Format: DXGI_FORMAT(16), InputSlot: 2, AlignedByteOffset: u32::MAX, InputSlotClass: D3D12_INPUT_CLASSIFICATION(1), InstanceDataStepRate: 7 },
        ];
        let mut d = described();
        d.InputLayout = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: 2 };
        let shaders = [11, 22, 0, 0, 55];
        let payload = unsafe { encode(0xabcd_ef01, shaders, &d) };
        assert!(payload.len() < 400, "{} bytes", payload.len());
        let r = pipeline(&payload);
        assert_eq!((r.key, r.shaders), (0xabcd_ef01, shaders));
        let (vs, ps, gs) = (b"vs".as_slice(), b"pixel".as_slice(), b"g".as_slice());
        let back = r.desc([Some(vs), Some(ps), None, None, Some(gs)]);
        let key = |d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC| unsafe { super::super::real::graphics_blob_key(d, b"dxil", None, None, &[]) };
        assert_eq!(key(&back), key(&d), "the state and the input layout");
        assert_ne!(key(&back), key(&described()), "(the input layout is in the key)");
        assert_eq!(unsafe { encode(0xabcd_ef01, shaders, &back) }, payload);
        assert_eq!((back.VS.BytecodeLength, back.PS.BytecodeLength, back.DS.BytecodeLength, back.GS.BytecodeLength), (2, 5, 0, 1));
        assert!(back.pRootSignature.is_none());
        // a description without an input layout
        let plain = unsafe { encode(1, [1, 0, 0, 0, 0], &described()) };
        let back = pipeline(&plain).desc([None; 5]);
        assert_eq!((back.InputLayout.NumElements, back.InputLayout.pInputElementDescs.is_null()), (0, true));
        assert_eq!(key(&back), key(&described()));
    }

    #[test]
    fn a_shader_entry_and_a_damaged_payload() {
        match decode(&shader_entry(0x77, b"code")) {
            Some(Entry::Shader(k, b)) => assert_eq!((k, b.as_slice()), (0x77, b"code".as_slice())),
            _ => panic!("not a shader entry"),
        }
        let payload = unsafe { encode(9, [9, 0, 0, 0, 0], &described()) };
        assert!(decode(&payload[..payload.len() - 1]).is_none(), "cut short");
        assert!(decode(&[payload.as_slice(), &[0]].concat()).is_none(), "a byte too many");
        assert!(decode(&[3, 0, 0, 0, 0, 0, 0, 0, 0]).is_none(), "another kind of entry");
    }

    /// Written, read back in order, each entry once; a file cut short in its last entry keeps the
    /// ones before it and the writer goes on after them.
    #[test]
    fn entries_are_written_once_and_read_back() {
        let dir = directory("round");
        let p = dir.0.join("cache").join("records.bin");
        let a = unsafe { encode(1, [1, 0, 0, 0, 0], &described()) };
        let b = shader_entry(2, b"shader");
        let mut w = Writer::open(&p, 1 << 20).unwrap();
        assert_eq!(w.add(&a).unwrap(), Added::New);
        assert_eq!(w.add(&b).unwrap(), Added::New);
        assert_eq!(w.add(&a).unwrap(), Added::Known);
        drop(w);
        assert_eq!(read(&p), vec![a.clone(), b.clone()]);
        let mut w = Writer::open(&p, 1 << 20).unwrap();
        assert_eq!((w.count(), w.add(&b).unwrap()), (2, Added::Known), "known from the file");
        drop(w);
        // a run cut short in its third entry
        let whole = std::fs::read(&p).unwrap();
        std::fs::write(&p, [whole.as_slice(), &[40, 0, 0, 0, 1, 2]].concat()).unwrap();
        assert_eq!(read(&p), vec![a.clone(), b.clone()]);
        let c = shader_entry(3, b"another");
        let mut w = Writer::open(&p, 1 << 20).unwrap();
        assert_eq!(w.add(&c).unwrap(), Added::New);
        drop(w);
        assert_eq!(read(&p), vec![a.clone(), b.clone(), c.clone()]);
        // a damaged entry ends the file there
        let mut bytes = std::fs::read(&p).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&p, &bytes).unwrap();
        assert_eq!(read(&p), vec![a, b]);
    }

    /// A file of another version (or no file of records at all) is read as none, and the writer
    /// starts it again.
    #[test]
    fn a_file_of_another_version_is_passed_over() {
        let dir = directory("version");
        let p = dir.0.join("records.bin");
        let a = shader_entry(5, b"five");
        let mut w = Writer::open(&p, 1 << 20).unwrap();
        w.add(&a).unwrap();
        drop(w);
        let mut bytes = std::fs::read(&p).unwrap();
        bytes[8..12].copy_from_slice(&(VERSION + 1).to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();
        assert!(read(&p).is_empty());
        let mut w = Writer::open(&p, 1 << 20).unwrap();
        assert_eq!((w.count(), w.add(&a).unwrap()), (0, Added::New));
        drop(w);
        assert_eq!(read(&p), vec![a.clone()]);
        assert_eq!(&std::fs::read(&p).unwrap()[8..12], &VERSION.to_le_bytes());
        std::fs::write(&p, b"not records").unwrap();
        assert!(read(&p).is_empty());
        assert!(read(&dir.0.join("none.bin")).is_empty());
    }

    /// An entry that would take the file past its cap is not written, nor any after it.
    #[test]
    fn the_file_stops_at_its_cap() {
        assert_eq!(cap_of(None), 64 << 20);
        assert_eq!(cap_of(Some(" 2 ")), 2 << 20);
        assert_eq!(cap_of(Some("lots")), 64 << 20);
        let dir = directory("cap");
        let p = dir.0.join("records.bin");
        let (a, b, small) = (shader_entry(1, &[1; 40]), shader_entry(2, &[2; 40]), shader_entry(3, &[]));
        // (room for `small` after `a`, not for `b`)
        let cap = (HEADER + 8 + a.len() + 8 + small.len()) as u64;
        let mut w = Writer::open(&p, cap).unwrap();
        assert_eq!(w.add(&a).unwrap(), Added::New);
        assert_eq!(w.add(&b).unwrap(), Added::Full);
        assert_eq!(w.add(&small).unwrap(), Added::Full, "nothing after the cap was hit");
        assert_eq!(w.add(&a).unwrap(), Added::Known);
        drop(w);
        assert_eq!(read(&p), vec![a]);
        assert!(std::fs::metadata(&p).unwrap().len() <= cap);
    }

    #[test]
    fn varints_round_trip() {
        for v in [0u64, 1, 0x7f, 0x80, 0x3fff, 0x4000, u32::MAX as u64, u64::MAX] {
            let mut out = Vec::new();
            put_var(&mut out, v);
            let mut r = Reader(&out);
            assert_eq!(r.var(), Some(v));
            assert!(r.0.is_empty());
        }
        assert_eq!(Reader(&[0x80]).var(), None, "cut short");
    }
}
