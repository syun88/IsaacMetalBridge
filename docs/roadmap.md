# Roadmap

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## M0 — Foundation (completed)

- pinned Apple source and source-grounded integration analysis
- protocol v1 control plane
- Swift Metal capability host
- C++20 guest probe and local integration tests
- honest environment doctor/setup scripts

## M1 — Real guest and Isaac Sim failure capture (completed)

- Apple `container` 1.1.0 is installed outside the repository
- a real guest reports `aarch64`
- the user-managed official Isaac Sim 6.0.1 ARM64 image is loaded outside Git
- NVIDIA's real compatibility checker ran and returned `System checking result: FAILED`
- captured failures include missing `libcuda.so.1`, NVML, `nvidia-smi`, and Vulkan `ERROR_INCOMPATIBLE_DRIVER`

The original driver-boundary failure was used as the trace baseline for the startup compatibility work below.

## M2 — Production control transport (implemented and verified)

- Linux AF_VSOCK guest listener on an explicit port
- macOS adapter using Apple container's verified `ContainerClient.dial`
- protocol 1.10 session verified across a real Apple-container ARM64 VM
- bounded framing and disconnect cleanup

Remaining hardening: fuzzing, connection timeout/cancellation, and an authentication threat model before any non-local use.

## M3 — Vulkan-to-Metal minimum compute path (prototype verified)

- completed below Vulkan: real `MTLBuffer` allocation, bounded upload/readback, one compute dispatch, and real command-buffer fence
- verified deterministic result: `[1,2,3,4] + 5 = [6,7,8,9]`, including across VM vsock
- completed discovery slice: Khronos Linux loader loads the IMB ICD, negotiates via vsock, and enumerates one Vulkan 1.0 physical/logical device and queue
- completed compute slice: Vulkan buffer/memory, descriptors, one exactly recognized `ADD_U32` SPIR-V module, compute pipeline, command buffer, queue submission, Vulkan fence, and readback map to the verified IMB/Metal path
- verified through the real Linux Vulkan loader and Apple-container VM: `[1,2,3,4] + 5 = [6,7,8,9]`
- completed startup compatibility additions include Kit-facing resources, descriptor indexing, timeline semaphores, query pools, and safe generic graphics submission while preserving the exact Metal compute path
- injected into the real Isaac Sim 6.0.1 ARM64 image and verified through the official base experience

The exact fixed-triangle rendering slice now produces and checks an actual image; this does not imply general or Isaac/RTX rendering.

## M4 — Rendering path

- completed: option-free RGBA8 image/texture mapping and full-image readback
- completed: exact build-generated vertex/fragment SPIR-V recognition
- completed: Vulkan render pass, framebuffer, fixed graphics pipeline, draw, queue submission, real Metal fence, pixel validation, and headless PNG output
- completed: exact tested Isaac Kit UI pipeline recognition, ring-buffer uploads, indexed draws, vertex offsets, scissor rectangles, BGRA8 texture sampling, and alpha blending
- completed: continuously captured real Kit UI frames displayed by a native macOS viewer
- completed: native pointer and keyboard input forwarding through Kit's real Carbonite input provider
- completed: pinned SPIRV-Cross tool and protocol 1.10 live compute-pipeline creation; 162 of 204 real Isaac requests compile on M4
- completed: generic Metal compute dispatch with push constants, buffer/image descriptors, and Vulkan-format texel-buffer views; the Full trace executes supported pipelines and safely skips unsupported/partially-bound RTX submissions
- completed: KHR triangle BLAS and instance TLAS compatibility objects backed by real Metal acceleration structures
- completed: real Metal ray dispatch against real Kit scene BLAS geometry and synchronization into Kit's triple-buffered 1280×720 viewport images
- completed: non-black `--demo-scene` viewport inside the live Kit UI, with mouse menu interaction and Stage-search keyboard input verified end to end
- completed: early `omni.usd` startup-stage extension replaces Isaac Sim base's disabled positional-USD path; local demo and real NVIDIA `Simple Grid` roots are verified by their `/World` children
- completed: unsupported NVIDIA/null scene TLAS records defer to a Metal identity-transform TLAS built from the completed real BLASes instead of aborting the queue submission
- completed: Vulkan sparse-image memory requirements, mip storage, and tile map/unmap mirrored into real Metal sparse heaps and enabled by default
- completed: real R32_UINT texel-buffer dispatch plus buffer/image descriptor and readback transport
- next: lower the 42 FP64-buffer compute requests while preserving Vulkan ABI and expand unsupported descriptor/format coverage
- next: preserve general scene-instance transforms and bridge live Kit camera state
- next: translate hit/miss shader semantics, materials/lighting, broader raster commands, and RTX sensor outputs

## M5 — CUDA startup compatibility (prototype startup slice completed)

- traced real CUDA Driver API and private export-table calls from Kit, PhysX, and Warp startup
- implemented the tested identity, context, allocation, copy, stream, event, module, and export-table startup surface
- Warp discovers one CUDA compatibility device during real Isaac Sim startup
- next: evaluate PTX translation and Metal execution feasibility and legal/technical constraints
- explicitly gate unsupported APIs

## M6 — Real Isaac Sim startup (Base and Full profiles completed)

- official `isaacsim.exp.base.kit` experience reports `app ready`
- verified both finite smoke-test shutdown and continued running state after readiness; the launcher now propagates a later container-init crash instead of treating readiness alone as success
- repeatable lifecycle is implemented in `scripts/run-isaac-sim.sh`
- targeted ARM virtual-counter compatibility removes the Apple-VM backwards-TSC abort
- official Full experience reports `Isaac Sim Full App is loaded`, reaches `app ready`, displays the real UI and Simple Grid scene presentation, and exits zero in finite validation
- stable Full startup isolates the bundled ROS environment; explicit `--ros2` remains experimental

GPU PhysX correctness, general RTX/Hydra shader/material/camera/sensor equivalence, long-running Full workload stability, and general CUDA compatibility remain research questions, not scheduled promises. The completed scene-validation milestone displays real loaded geometry through Metal, but currently uses a fallback scene-instance/camera/shading path.
