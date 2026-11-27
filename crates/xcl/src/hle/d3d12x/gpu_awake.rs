//! AMD GPU keep-awake compute heartbeat (`XCL_GPU_AWAKE`).
//!
//! On AMD RDNA2/RDNA3 GPUs, the SMU can drop shader clocks during 1.2–2.4 ms CPU gaps between
//! title command-list submissions, inflating GPU frame time by 20–30%. While the title is actively
//! submitting GPU work (within the last 250 ms), a single lightweight background thread keeps at
//! most two tiny async-compute dispatches in flight on a dedicated `D3D12_COMMAND_LIST_TYPE_COMPUTE`
//! queue so the GPU never falls into deep idle between submits, and sleeps immediately whenever
//! title submission pauses (menus, shader compile stalls, idle).
//!
//! Controlled by `XCL_GPU_AWAKE`:
//! - `0` / `off` / `false`: disabled
//! - `1` / `on` / `true`: forced on
//! - unset / `auto` / other: enabled automatically on AMD adapters (`vendor_id == 0x1002`)

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

/// PCI vendor ID for AMD GPUs.
const AMD_VENDOR_ID: u32 = 0x1002;
/// How long after the most recent title GPU submit we keep submitting heartbeats.
const ACTIVE_WINDOW_MS: u64 = 250;
/// Sleep interval while the title is idle (no submit within `ACTIVE_WINDOW_MS`).
const IDLE_SLEEP_MS: u64 = 4;
/// Fence wait slice so `stop()` or device removal is noticed promptly.
const WAIT_SLICE_MS: u32 = 100;
/// Bounded drain timeout on clean shutdown before dropping D3D12 objects.
const SHUTDOWN_DRAIN_MS: u32 = 500;

/// Minimal SM 5.0 compute shader: 64 threads running a 400,000-step LCG and writing 1 uint each
/// (256 bytes total) to a root UAV at `u0` (no descriptor heap required).
const HEARTBEAT_HLSL: &str = r#"
RWByteAddressBuffer dst : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x ^ 0x9e3779b9u;
    [loop]
    for (uint i = 0u; i < 400000u; ++i) {
        x = x * 1664525u + 1013904223u;
    }
    dst.Store(tid.x * 4u, x);
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Off,
    On,
    Auto,
}

fn parse_mode(val: Option<&str>) -> Mode {
    let Some(raw) = val.map(str::trim).filter(|s| !s.is_empty()) else {
        return Mode::Auto;
    };
    if raw.eq_ignore_ascii_case("0") || raw.eq_ignore_ascii_case("off") || raw.eq_ignore_ascii_case("false") {
        Mode::Off
    } else if raw.eq_ignore_ascii_case("1") || raw.eq_ignore_ascii_case("on") || raw.eq_ignore_ascii_case("true") {
        Mode::On
    } else {
        Mode::Auto
    }
}

fn should_enable(mode: Mode, vendor_id: u32) -> bool {
    match mode {
        Mode::Off => false,
        Mode::On => true,
        Mode::Auto => vendor_id == AMD_VENDOR_ID,
    }
}

fn configured_mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var("XCL_GPU_AWAKE").ok().as_deref()))
}

static STARTED: OnceLock<()> = OnceLock::new();
static STOP: AtomicBool = AtomicBool::new(false);
static LAST_SUBMIT_MS: AtomicU64 = AtomicU64::new(0);

fn epoch_ms() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    // +1 so 0 unambiguously means "no title submit observed yet".
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// Record that the title just submitted GPU work so the heartbeat thread stays active for the next
/// `ACTIVE_WINDOW_MS` (250 ms). Lock-free single atomic store on the hot submit path.
pub(super) fn note_activity() {
    LAST_SUBMIT_MS.store(epoch_ms(), Ordering::Relaxed);
}

/// Signal the background heartbeat thread to stop cleanly.
pub(super) fn stop() {
    STOP.store(true, Ordering::Release);
}

/// Start the GPU keep-awake heartbeat thread if enabled by `XCL_GPU_AWAKE` (auto-enabled on AMD
/// `vendor_id == 0x1002`). Idempotent via `OnceLock`.
///
/// # Safety
/// `device` must be a valid live `ID3D12Device`.
pub(super) unsafe fn start(device: &ID3D12Device, vendor_id: u32) {
    let mode = configured_mode();
    if !should_enable(mode, vendor_id) {
        return;
    }
    let dev_clone = device.clone();
    STARTED.get_or_init(|| {
        STOP.store(false, Ordering::Release);
        struct SendDevice(ID3D12Device);
        // COM ID3D12Device methods used here are free-threaded per D3D12 spec.
        unsafe impl Send for SendDevice {}
        let send_dev = SendDevice(dev_clone);
        let spawned = std::thread::Builder::new().name("xcl-gpu-awake".into()).spawn(move || {
            let dev = send_dev;
            run_worker(dev.0, vendor_id);
        });
        if let Err(e) = spawned {
            crate::xlog_warn!("D3D12.X: failed to spawn xcl-gpu-awake thread: {e}");
        }
    });
}

fn compile_heartbeat_cs() -> Option<Vec<u8>> {
    unsafe {
        let mut blob = None;
        let r = D3DCompile(
            HEARTBEAT_HLSL.as_ptr() as *const c_void,
            HEARTBEAT_HLSL.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(b"main\0".as_ptr()),
            PCSTR(b"cs_5_0\0".as_ptr()),
            0,
            0,
            &mut blob,
            None,
        );
        match (r, blob) {
            (Ok(()), Some(b)) => Some(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize()).to_vec()),
            _ => None,
        }
    }
}

struct EventGuard(windows_sys::Win32::Foundation::HANDLE);

impl EventGuard {
    fn new() -> Result<Self, String> {
        let h = unsafe { windows_sys::Win32::System::Threading::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if h.is_null() {
            Err("CreateEventW failed".into())
        } else {
            Ok(Self(h))
        }
    }

    fn as_win_handle(&self) -> HANDLE {
        HANDLE(self.0)
    }

    fn wait(&self, timeout_ms: u32) {
        unsafe {
            windows_sys::Win32::System::Threading::WaitForSingleObject(self.0, timeout_ms);
        }
    }
}

impl Drop for EventGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.0);
            }
        }
    }
}

struct RecordedSlot {
    _alloc: ID3D12CommandAllocator,
    _list: ID3D12GraphicsCommandList,
    cmd_list: ID3D12CommandList,
}

struct HeartbeatState {
    queue: ID3D12CommandQueue,
    _uav_buf: ID3D12Resource,
    _root_sig: ID3D12RootSignature,
    _pso: ID3D12PipelineState,
    slots: [RecordedSlot; 2],
    fence: ID3D12Fence,
    event: EventGuard,
}

unsafe fn init_state(device: &ID3D12Device) -> Result<HeartbeatState, String> {
    let bytecode = compile_heartbeat_cs().ok_or_else(|| "D3DCompile failed for heartbeat compute shader".to_string())?;

    let queue: ID3D12CommandQueue = device
        .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        })
        .map_err(|e| format!("CreateCommandQueue(COMPUTE) failed: {e}"))?;

    let mut uav_opt: Option<ID3D12Resource> = None;
    device
        .CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut uav_opt,
        )
        .map_err(|e| format!("CreateCommittedResource(UAV) failed: {e}"))?;
    let uav_buf = uav_opt.ok_or_else(|| "CreateCommittedResource(UAV) returned null".to_string())?;
    let uav_va = uav_buf.GetGPUVirtualAddress();

    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &param,
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut rs_blob = None;
    D3D12SerializeRootSignature(&rs_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut rs_blob, None)
        .map_err(|e| format!("D3D12SerializeRootSignature failed: {e}"))?;
    let rs_blob = rs_blob.ok_or_else(|| "D3D12SerializeRootSignature returned null blob".to_string())?;
    let rs_bytes = std::slice::from_raw_parts(rs_blob.GetBufferPointer() as *const u8, rs_blob.GetBufferSize());
    let root_sig: ID3D12RootSignature = device
        .CreateRootSignature(0, rs_bytes)
        .map_err(|e| format!("CreateRootSignature failed: {e}"))?;

    let mut pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytecode.as_ptr() as *const c_void,
            BytecodeLength: bytecode.len(),
        },
        ..Default::default()
    };
    let pso_res: windows::core::Result<ID3D12PipelineState> = device.CreateComputePipelineState(&pso_desc);
    ManuallyDrop::drop(&mut pso_desc.pRootSignature);
    let pso = pso_res.map_err(|e| format!("CreateComputePipelineState failed: {e}"))?;

    let make_slot = || -> Result<RecordedSlot, String> {
        let alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
            .map_err(|e| format!("CreateCommandAllocator(COMPUTE) failed: {e}"))?;
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &alloc, Some(&pso))
            .map_err(|e| format!("CreateCommandList(COMPUTE) failed: {e}"))?;
        list.SetComputeRootSignature(&root_sig);
        list.SetPipelineState(&pso);
        list.SetComputeRootUnorderedAccessView(0, uav_va);
        list.Dispatch(1, 1, 1);
        list.Close().map_err(|e| format!("CommandList::Close failed: {e}"))?;
        let cmd_list: ID3D12CommandList = list.cast().map_err(|e| format!("cast to ID3D12CommandList failed: {e}"))?;
        Ok(RecordedSlot { _alloc: alloc, _list: list, cmd_list })
    };
    let slots = [make_slot()?, make_slot()?];

    let fence: ID3D12Fence = device
        .CreateFence(0, D3D12_FENCE_FLAG_NONE)
        .map_err(|e| format!("CreateFence failed: {e}"))?;
    let event = EventGuard::new()?;

    Ok(HeartbeatState {
        queue,
        _uav_buf: uav_buf,
        _root_sig: root_sig,
        _pso: pso,
        slots,
        fence,
        event,
    })
}

fn run_worker(device: ID3D12Device, vendor_id: u32) {
    let state = match super::title_textures::own_work(|| unsafe { init_state(&device) }) {
        Ok(s) => s,
        Err(e) => {
            crate::xlog_warn!("D3D12.X: GPU keep-awake heartbeat disabled ({e})");
            return;
        }
    };
    crate::xlog_info!("D3D12.X: GPU keep-awake async-compute heartbeat active (vendor {vendor_id:#06x}, XCL_GPU_AWAKE=0 to disable)");

    let mut fence_val: u64 = 0;
    while !STOP.load(Ordering::Acquire) {
        if unsafe { device.GetDeviceRemovedReason() }.is_err() {
            break;
        }
        let last = LAST_SUBMIT_MS.load(Ordering::Relaxed);
        let now = epoch_ms();
        if last == 0 || now.saturating_sub(last) > ACTIVE_WINDOW_MS {
            std::thread::sleep(Duration::from_millis(IDLE_SLEEP_MS));
            continue;
        }

        // Keep at most 2 dispatches in flight: wait until the slot we are about to reuse has
        // finished executing on the GPU (`fence >= fence_val - 1`).
        let wait_target = fence_val.saturating_sub(1);
        if wait_target > 0 {
            loop {
                let completed = unsafe { state.fence.GetCompletedValue() };
                if completed == u64::MAX || unsafe { device.GetDeviceRemovedReason() }.is_err() {
                    return;
                }
                if completed >= wait_target || STOP.load(Ordering::Acquire) {
                    break;
                }
                if unsafe { state.fence.SetEventOnCompletion(wait_target, state.event.as_win_handle()) }.is_err() {
                    return;
                }
                state.event.wait(WAIT_SLICE_MS);
            }
            if STOP.load(Ordering::Acquire) {
                break;
            }
        }

        let slot_idx = (fence_val as usize) & 1;
        let next_val = fence_val + 1;
        let submitted = super::title_textures::own_work(|| unsafe {
            state.queue.ExecuteCommandLists(&[Some(state.slots[slot_idx].cmd_list.clone())]);
            state.queue.Signal(&state.fence, next_val).is_ok()
        });
        if !submitted {
            break;
        }
        fence_val = next_val;
    }

    // Drain in-flight heartbeats briefly on clean stop so GPU finishes before COM objects drop.
    if fence_val > 0 && unsafe { device.GetDeviceRemovedReason() }.is_ok() {
        let completed = unsafe { state.fence.GetCompletedValue() };
        if completed < fence_val && completed != u64::MAX {
            if unsafe { state.fence.SetEventOnCompletion(fence_val, state.event.as_win_handle()) }.is_ok() {
                state.event.wait(SHUTDOWN_DRAIN_MS);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_mode_parsing_and_amd_auto_enable() {
        assert_eq!(parse_mode(None), Mode::Auto);
        assert_eq!(parse_mode(Some("")), Mode::Auto);
        assert_eq!(parse_mode(Some("auto")), Mode::Auto);
        assert_eq!(parse_mode(Some("unknown")), Mode::Auto);

        for off in ["0", "off", "OFF", "false", "False", " 0 "] {
            assert_eq!(parse_mode(Some(off)), Mode::Off, "expected Off for {off:?}");
        }
        for on in ["1", "on", "ON", "true", "True", " 1 "] {
            assert_eq!(parse_mode(Some(on)), Mode::On, "expected On for {on:?}");
        }

        // Auto enables only for AMD (0x1002); On/Off override regardless of vendor.
        assert!(should_enable(Mode::Auto, AMD_VENDOR_ID));
        assert!(!should_enable(Mode::Auto, 0x10DE)); // NVIDIA
        assert!(!should_enable(Mode::Auto, 0x8086)); // Intel
        assert!(!should_enable(Mode::Off, AMD_VENDOR_ID));
        assert!(should_enable(Mode::On, 0x10DE));
        assert!(should_enable(Mode::On, 0x8086));
    }

    #[test]
    fn heartbeat_hlsl_compiles_with_d3dcompile() {
        let bc = compile_heartbeat_cs().expect("HEARTBEAT_HLSL failed D3DCompile(cs_5_0)");
        assert!(bc.len() >= 32, "expected valid DXBC container, got {} bytes", bc.len());
        assert_eq!(&bc[..4], b"DXBC", "expected DXBC magic header");
    }
}
