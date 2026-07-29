# Feasibility matrix

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

Classifications describe the current repository at the audited revisions. “Architecturally possible” is not a working implementation claim.

| Capability | Classification | Evidence / missing work |
|---|---|---|
| Linux ARM64 Isaac Sim 6.0.1 package/image | Verified startup profile | Official ARM64 image is loaded; the real compatibility checker and official base experience both reached `app ready` under Apple `container`. |
| Apple `container` ARM64 Linux VM | Verified available | Apple `container` 1.1.0 is installed; its service is running and a real guest returned `aarch64`. |
| CPU PhysX | Startup verified, results unvalidated | The real PhysX extensions start in the base experience; simulation correctness still needs workload-level validation. |
| Vulkan loader/device discovery | Verified prototype only | Khronos Linux loader and real Kit load the IMB manifest/library and enumerate `IsaacMetalBridge NVIDIA-compat (Apple M4)`. The ICD remains non-conformant. |
| Vulkan compute commands | Verified narrow prototype | Standard Vulkan buffer/memory, descriptor, one exactly recognized SPIR-V compute pipeline, command buffer, queue submission, fence, and readback produced `[6,7,8,9]` through vsock and real Metal. |
| Vulkan raster commands | Verified narrow prototypes | The fixed RGBA8 pipeline produced a checked 64×64 triangle PNG. The exact observed Kit UI pipeline also produces real application chrome through Metal. General graphics remain absent. |
| Vulkan ray tracing | Verified narrow compatibility slice | The probe and real Isaac trace build KHR BLAS/TLAS compatibility objects backed by Metal and dispatch real intersection rays. General Vulkan RT shader semantics are not implemented. |
| Vulkan sparse images | Safety-gated, not implemented | Sparse buffers remain available, but sparse image residency is hidden by default. NVIDIA `Simple Grid` geometry now loads and dispatches safely from startup; its three source PNGs resolve, but the texture processor cannot make them resident, so the familiar wireframe material remains incomplete. |
| Metal ray tracing | Verified execution on current host | Real primitive/instance acceleration structures and a 64×64 probe plus 1280×720 real-Isaac scene dispatch execute on Apple M4. |
| CUDA Driver API | Verified startup prototype | The tested Driver API/private export-table subset lets Kit, PhysX startup, and Warp discover one compatibility device; general kernel execution is absent. |
| PTX conversion | Unknown | Toolchain, licensing, correctness, and feature coverage need research. |
| CUDA streams | Startup handles only | Stream create/query/synchronize/destroy calls are implemented for startup, without general CUDA ordering or kernel semantics. |
| CUDA events | Startup handles only | Event create/record/query/synchronize/destroy calls are implemented for startup, without CUDA timestamp semantics. |
| NVML | Requires Linux guest modification | A compatibility library could answer a subset, but truthful telemetry mapping is undecided. |
| GPU PhysX | Blocked by NVIDIA private technology | Depends on NVIDIA CUDA/GPU implementation details not exposed by this project. |
| Warp | Device discovery verified | Warp 1.13.0 reports the IMB CUDA compatibility device; arbitrary Warp kernels are not validated. |
| Isaac Lab | Unknown | Depends on Isaac Sim, Warp, PyTorch/CUDA, memory capacity, and training workloads. |
| RTX Camera | Blocked by NVIDIA private technology | RTX renderer and sensor pipeline are not public Metal implementations. |
| RTX LiDAR | Blocked by NVIDIA private technology | Requires NVIDIA RTX sensor/rendering behavior. |
| GUI | Interactive chrome and validation viewport verified | A native macOS viewer shows the real Kit menus, panels, toolbar, Stage, Property, and Content UI, forwards pointer/keyboard input through Carbonite, and composites a non-black real-geometry Metal intersection result for the included validation stage. |
| WebRTC | Out of scope | Requires a working renderer and networking validation first. |
| Host/guest vsock | Verified available | `imb-container-host` uses Apple's `ContainerClient.dial`; a real ARM64 guest completed IMB 1.7 Metal compute, SPIR-V pipeline creation, fixed/UI raster, acceleration structures, ray dispatch, and final frame delivery over the returned full-duplex file handle. |
| Metal buffer/compute primitive | Verified available | Real `MTLBuffer`, upload/readback, `ADD_U32` compute, and command-buffer fence produced `[6,7,8,9]` from `[1,2,3,4] + 5` on Apple M4, including through a standard Vulkan dispatch. |
| Dedicated VirtIO GPU | Requires Apple container modification | No GPU device is configured in audited source; VZ extension wiring is needed and VZ support must exist. |
| Shared-memory data plane | Unknown | No verified coherent shared-memory mechanism has been selected. |
