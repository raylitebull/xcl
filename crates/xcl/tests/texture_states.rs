//! Uploads into a texture whose subresources are in different states (#104): the barriers the
//! uploads of title textures record start from each subresource's own state and return to it, for
//! all of a texture and for parts of it, checked by the D3D12 debug layer's state tracking. Its own
//! process: the debug layer has to be on before the first device exists.

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

use xcl::gpu::texture;
use xcl::hle::d3d12x::title_textures;
use xcl::hle::xg;

const SRV: D3D12_RESOURCE_STATES = texture::SHADER_READ;

/// The debug layer's error and corruption messages since the last call (then cleared).
unsafe fn errors(q: &ID3D12InfoQueue) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..q.GetNumStoredMessages() {
        let mut len = 0usize;
        if q.GetMessage(i, None, &mut len).is_err() || len == 0 {
            continue;
        }
        let mut buf = vec![0u64; len.div_ceil(8)];
        let m = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        if q.GetMessage(i, Some(m), &mut len).is_ok() && matches!((*m).Severity, D3D12_MESSAGE_SEVERITY_ERROR | D3D12_MESSAGE_SEVERITY_CORRUPTION) {
            let d = std::slice::from_raw_parts((*m).pDescription as *const u8, (*m).DescriptionByteLength.saturating_sub(1));
            out.push(String::from_utf8_lossy(d).into_owned());
        }
    }
    q.ClearStoredMessages();
    out
}

unsafe fn transition(r: &ID3D12Resource, sub: u32, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: sub, StateBefore: before, StateAfter: after }),
        },
    }
}

/// A texture in shader-resource states with mip 1 moved to COPY_SOURCE (as a title moves one
/// mip to copy from it), then uploaded again: with each subresource's state no error; with the
/// creation state for all of them (what uploads used before), the debug layer sees the
/// mismatch – so the check is a real one.
#[test]
fn uploads_start_from_each_subresources_state() {
    unsafe {
        use std::io::Write;
        match xcl::gpu::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
            Ok(d) => d.EnableDebugLayer(),
            Err(e) => {
                let _ = writeln!(std::io::stderr(), "SKIPPED uploads_start_from_each_subresources_state: no D3D12 debug layer ({e})");
                return;
            }
        }
        let Ok(dev) = xcl::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
            xcl::gpu::recomp::runner::skip_test("no GPU");
            return;
        };
        let Ok(q) = dev.cast::<ID3D12InfoQueue>() else {
            let _ = writeln!(std::io::stderr(), "SKIPPED uploads_start_from_each_subresources_state: the device has no debug layer");
            return;
        };
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        let mips = 4u16;
        let rd = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).unwrap();
        let r = r.unwrap();
        // the title's description of it (4KB_S) and its memory
        let mut desc = [0u8; 0x38];
        desc[0..4].copy_from_slice(&3u32.to_le_bytes());
        desc[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        desc[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        desc[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
        desc[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
        desc[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        desc[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        desc[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = xg::Texture::new(&desc).unwrap();
        let size = tex.size() as usize;
        let chain = &tex.texels().unwrap().chain;

        // made in COPY_DEST, left in the shader-resource states
        let up = |states: Vec<D3D12_RESOURCE_STATES>, v: u8| title_textures::upload_and_wait(&dev, &queue, &r, &desc, &vec![v; size], states, None).unwrap();
        // a part of the top level (the blocks of its last rows) alone
        let top = chain.level_bytes(0);
        let part = |states: Vec<D3D12_RESOURCE_STATES>, v: u8| title_textures::upload_and_wait(&dev, &queue, &r, &desc, &vec![v; size], states, Some(vec![(top.end - 4096, top.end)])).unwrap();
        up(vec![D3D12_RESOURCE_STATE_COPY_DEST], 1);
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
        list.ResourceBarrier(&[transition(&r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, SRV)]);
        list.ResourceBarrier(&[transition(&r, 1, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        texture::wait(&dev, &queue).unwrap();
        let setup = errors(&q);
        assert!(setup.is_empty(), "setup: {setup:?}");

        // each subresource's state: no error, and it is left in it
        let mut states = vec![SRV; mips as usize];
        states[1] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        up(states.clone(), 2);
        let right = errors(&q);
        println!("per-subresource states: {} debug-layer errors", right.len());
        assert!(right.is_empty(), "uploads with each subresource's state: {right:?}");
        up(states.clone(), 3);
        assert!(errors(&q).is_empty(), "the upload did not leave the subresources in their states");
        // part of the top level: that subresource's state alone (mip 1 is in another one)
        part(states.clone(), 4);
        part(vec![SRV], 5);
        let of_part = errors(&q);
        assert!(of_part.is_empty(), "an upload of a part of one subresource: {of_part:?}");

        // one state for all of them, as uploads did before: mip 1 is not in it
        up(vec![SRV], 6);
        let wrong = errors(&q);
        println!("one state for all: {} debug-layer errors, first: {:?}", wrong.len(), wrong.first());
        assert!(!wrong.is_empty(), "the debug layer did not see a transition from the wrong state – the check above proves nothing");
    }
}
