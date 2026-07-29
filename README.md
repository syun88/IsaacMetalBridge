# IsaacMetalBridge

## Disclaimer

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

This is not an official Apple project or an official NVIDIA Isaac Sim port. There is no cooperation with or endorsement from Apple or NVIDIA. The project does not use Apple or NVIDIA logos and must not be presented as official.

## Overview

IsaacMetalBridge investigates whether the real Linux ARM64 build of NVIDIA Isaac Sim can retain its Linux, Omniverse Kit, Carbonite, PhysX, USD, and Python environment while GPU work is bridged to Metal on an Apple Silicon Mac.

It is a compatibility-layer research project, not a simulator, an Isaac Sim clone, or an API reimplementation. MuJoCo and replacement simulators are explicitly out of scope.

## Motivation

Isaac Sim 6.0.1 is available for Linux aarch64, but NVIDIA's supported aarch64 target is DGX Spark with an NVIDIA driver and GPU. Apple Silicon has a different GPU, driver model, shader toolchain, and ray-tracing interface. CPU instruction-set compatibility therefore does not provide GPU compatibility.

## Why compatibility layer approach

Preserving the Linux ARM64 user space keeps the real Isaac Sim distribution and its internal software stack intact. Apple `container` supplies a native Apple Silicon Linux VM boundary; IsaacMetalBridge focuses only on the missing GPU and host/guest interoperability layers.

## Architecture

```text
Linux ARM64 Isaac Sim
  -> guest Vulkan/CUDA compatibility layers
  -> versioned IMB protocol
  -> verified transport (prototype: pipes; Apple integration: host-dialed vsock)
  -> macOS Swift host bridge
  -> Metal
```

See [docs/architecture.md](docs/architecture.md) for the detailed execution, GPU, resource, and rendering paths.

## Current Status

Real Isaac Sim base-experience startup and its Kit UI through Metal:

- Apple `container` is pinned as a clean submodule.
- Its VM, XPC, OCI-image, VirtIO, and vsock paths have been inspected from source.
- A versioned binary control protocol is defined.
- A Swift host reports real Metal capabilities and owns real shared `MTLBuffer` resources.
- A C++20 Linux ARM64 guest listens on AF_VSOCK; the macOS adapter dials it through Apple's public `ContainerClient.dial(id:port:)` API.
- Protocol 1.10 uploads bytes, runs real Metal compute, creates linear and sparse Metal textures, mirrors Vulkan sparse-image tile map/unmap operations into Metal sparse heaps, draws the tested raster/UI streams, creates translated Metal compute pipelines from Vulkan SPIR-V, transports buffer, image, and texel-buffer descriptors, builds Metal primitive/instance acceleration structures, and dispatches Metal rays.
- The deterministic `[1,2,3,4] + 5 = [6,7,8,9]` path is verified both locally and across a real Apple-container Linux VM.
- A Linux ARM64 Vulkan loader discovers the experimental IMB device and completes one standard Vulkan compute flow through vsock and real Metal.
- A standard Vulkan image/render-pass/graphics-pipeline/`vkCmdDraw` flow produces a checked 64×64 RGBA8 triangle through vsock and real Metal; the PNG artifact is written under `build/artifacts`.
- The real Isaac Sim 6.0.1 ARM64 image loads the IMB Vulkan ICD and CUDA startup shim. Warp discovers `cuda:0` as the Apple-M4 compatibility device.
- The official `isaacsim.exp.base.kit` experience starts headlessly, loads Isaac core, PhysX, sensors, and `isaacsim.simulation_app`, then reports `app ready`.
- The tested Kit UI vertex/index rings, font texture, scissor rectangles, and alpha-blended draw lists are bridged to Metal. A native macOS viewer shows the live real Isaac Sim menus, panels, toolbar, Stage, Property, and Content UI with `--window`.
- The native viewer forwards pointer, scroll, and keyboard events into the real Kit `carb.input.InputProvider`. Real Create-menu interaction and Stage search-field typing are verified end to end.
- The KHR ray-tracing compatibility path records the real scene BLAS geometry, builds matching Metal primitive acceleration structures, supplies a narrow identity-transform TLAS fallback when Kit leaves its NVIDIA-only scene-instance descriptor null, and performs a real Metal intersection dispatch.
- The latest ray result is synchronized into Kit's actual triple-buffered 1280×720 viewport images before the final indexed UI pass. The included `--demo-scene` validation stage therefore displays a non-black real-geometry intersection image inside the live Kit application chrome.
- The launcher now waits for the real container init process and returns its status. A Kit crash after `app ready` is reported as a crash instead of being misreported as successful startup.
- Sparse buffer and sparse image residency are enabled by default. Vulkan sparse-image requirements, tile map/unmap operations, and mip storage are backed by real Metal sparse heaps; the real-container probe validates map and unmap.
- A pinned Khronos SPIRV-Cross build translates live Isaac compute modules on the host. In the current real-image trace, 162 of 204 requested compute pipelines reached `MTLComputePipelineState`; the remaining 42 contain FP64 buffer operations that require ABI-preserving lowering. Generic Vulkan compute dispatch now executes supported pipelines with buffer, storage-image, sampled-image, push-constant, and formatted texel-buffer bindings. A finite Full run measured 595 successful Metal dispatches while safely skipping 2,708 unsupported or partially bound RTX submissions.
- A targeted ARM virtual-counter compatibility shim removes the backwards-TSC startup abort. The official Full experience reaches `Isaac Sim Full App is loaded`, reports `app ready`, displays the real Isaac UI, and exits zero in finite validation.
- `--simple-grid` opens NVIDIA's real `default_environment.usd` in either Base or Full. The current Full viewer shows its Stage prims plus a visible blue wire grid and axes through the bridge's Metal scene-presentation path.
- This is not conformant Vulkan or CUDA. The visible scene path is still a bounded Metal raster/ray validation implementation, not full RTX/Hydra shader, material, lighting, transform, camera, or sensor equivalence. Arbitrary graphics shaders, general CUDA kernels, validated GPU PhysX results, OptiX/NGX, live RTX cameras, and RTX sensors remain unsupported.

## Roadmap

1. Completed: validate Apple `container`, ARM64 Linux, and the real Isaac Sim compatibility failure.
2. Completed: connect a Linux guest to the macOS host through Apple's existing host-dialed vsock API without patching the submodule.
3. Completed at IMB-protocol level: real Metal buffer upload, compute, fence, and readback.
4. Completed prototype slice: Vulkan buffer/memory, one exactly recognized SPIR-V compute pipeline, command submission, fence, and readback map to verified IMB behavior.
5. Completed startup slice: inject the ICD and CUDA shim into the real Isaac Sim ARM64 image and reach `app ready` in the official base experience.
6. Completed interactive UI slice: real Isaac Kit UI draws through Metal in a native macOS window and receives pointer/keyboard input.
7. Completed sparse/compute/full-startup slice: protocol 1.10 maps sparse Vulkan image tiles to Metal, executes supported general compute and texel-buffer dispatches, builds Metal BLAS/TLAS resources, and reaches `app ready` in the official Full experience. Next: broader shader semantics, scene/camera state, materials/lighting, and RTX sensors.

See [docs/roadmap.md](docs/roadmap.md).

## Supported Features

- IMB protocol version negotiation and message validation
- Host Metal device discovery
- Real shared Metal buffers, bounded upload/readback, `ADD_U32` compute, and real command-buffer fences
- Local pipe and real Apple-container host-to-guest vsock validation
- Experimental Vulkan loader/ICD discovery, 16 queues, storage-buffer and R32_UINT texel-buffer compute dispatches, transfer/image readback, and sparse-image tile residency through a real Linux ARM64 guest
- One exact Vulkan RGBA8 render pass and fixed triangle pipeline, with real Metal rasterization, fence completion, image readback, pixel validation, and PNG output
- One exactly recognized Isaac Kit UI pipeline with indexed draws, vertex offsets, scissor rectangles, BGRA8 textures, alpha blending, frame capture, and a native macOS viewer
- Native pointer, button, drag, scroll, and keyboard forwarding into the real Isaac Kit input devices
- Reproducible pinned SPIRV-Cross tooling, live SPIR-V-to-MSL compute-pipeline creation, and supported generic Vulkan compute dispatch on Apple M4
- Vulkan KHR BLAS/TLAS compatibility, real Metal primitive/instance acceleration structures, and real Metal ray dispatch
- Non-black 3D validation viewport for the included cube/ground stage and a visible wire-grid presentation for NVIDIA Simple Grid, composited into the real Kit UI
- Vulkan resource/synchronization compatibility needed for real Kit startup, including descriptor indexing, timeline semaphores, and query pools
- CUDA Driver startup identity, context, memory, stream, event, module, and private export-table compatibility sufficient for the tested extensions and Warp discovery
- Real Isaac Sim 6.0.1 official Base and Full experience startup to `app ready`
- Apple-container patch apply/reverse workflow
- Non-downloading Isaac Sim environment checks

## Unsupported Features

- Conformant or general-purpose Vulkan rendering, arbitrary graphics shaders, full RTX/Hydra shader/material/light/camera equivalence, or FP64 shader-buffer execution
- General CUDA/PTX kernel execution or NVML
- Validated GPU PhysX results, Isaac Lab, RTX Camera output, or RTX LiDAR output
- General USD scene transforms, interactive bridge-camera navigation, RTX sensor pixels, or WebRTC
- Complete streamed RTX environment materials across all Vulkan formats and descriptor patterns
- Shared-memory or bulk zero-copy transport
- NVIDIA-only OptiX, NGX, NVML, and `nvidia-smi`

## Development Setup

Requirements: Apple Silicon macOS, Swift 6+, CMake 3.25+, and a C++20 compiler.

```sh
./scripts/bootstrap.sh
./scripts/build-all.sh
./scripts/test-all.sh
```

Build the Apple `ContainerAPIClient` adapter and run the real Linux-VM/vsock/Metal test with:

```sh
./scripts/build-container-adapter.sh
./scripts/test-container-vsock.sh
./scripts/test-vulkan-icd.sh
# or run all local tests plus the real container test
./scripts/test-all.sh --container
```

`bootstrap.sh` initializes the pinned submodule but does not install packages or download Isaac Sim. Build output stays under ignored `.build`/`build` directories.

## Isaac Sim Environment

The validation baseline is real NVIDIA Isaac Sim 6.0.1 for Linux aarch64. The project never downloads it automatically and never stores its binaries, images, caches, credentials, assets, or license-controlled content in Git.

Set `ISAAC_SIM_PATH` to an existing external ARM64 installation, or `IMB_ISAAC_IMAGE` to an image reference already managed outside this repository, then run:

```sh
./scripts/setup-isaac-environment.sh
```

An ignored local `.env` may hold the non-secret image reference and platform. It must not contain registry credentials or license-acceptance values.

See [docs/isaac-sim-environment.md](docs/isaac-sim-environment.md).

After reviewing NVIDIA's terms and explicitly accepting the EULA, build and start the official Full profile with NVIDIA's Simple Grid in a native macOS viewer:

```sh
./scripts/build-all.sh
./scripts/build-container-adapter.sh
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid
```

The first run builds `imb-isaac-sim:6.0.1-dev` from the already-loaded real image. `--simple-grid` downloads and caches NVIDIA's real Simple Grid asset outside Git and opens it through an early Kit extension. Omit all scene flags for the normal empty stage; no diagnostic USD is inserted by default. `--demo-scene` remains an explicit bridge regression scene. With `--no-build`, the launcher automatically rebuilds only a missing small override and bind-mounts the current Vulkan/CUDA files, so a stale image-baked protocol cannot be paired with the current host. The viewer initially says it is starting, then displays the real Kit UI after frames arrive. Pointer and keyboard events inside the viewer are returned to Kit. On a 32 GiB Mac the launcher assigns 24 GiB to the guest by default, leaving 8 GiB for macOS and the viewer; set `IMB_ISAAC_MEMORY=20g` (or another safe value) to override it. Press Ctrl-C in Terminal to close the viewer and remove the run container. For a repeat launch without rebuilding, or a finite Full smoke test:

```sh
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid --no-build
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --no-build --quit-after 150
```

`--quit-after` is a frame count, not seconds. Success is the real Kit log line `app ready`, a zero container-init exit status, the launcher confirmation, and a window titled `Isaac Sim 6.0.1 — Apple Metal Bridge`. The launcher treats a nonzero exit after readiness as a crash. The visible application chrome and its menu/search responses are generated by the real Isaac Kit process, not a mock UI. With `--simple-grid`, the current bridge should show the wire grid/axes rather than a fully black center; with `--demo-scene`, it should show the bounded blue geometry-intersection view.

### Base, full, demo scene, and editor grid

The launcher defaults to the official `isaacsim.exp.base.kit` experience. `--experience full` selects `isaac-sim.sh`; a targeted virtual-counter shim handles the ARM VM timer path that previously aborted in `omni.anim.behavior.core`. Both profiles now reach `app ready`, and Full is the recommended profile when the complete Isaac application extension set is required.

`--demo-scene` selects [guest/scenes/metal-ray-scene.usda](guest/scenes/metal-ray-scene.usda). It is a local cube-and-ground validation scene, not NVIDIA's Simple Grid environment. Isaac Sim 6.0.1's base experience ignores a positional USD argument because its legacy `/app/content/usdFile` opener is disabled. The launcher therefore disables automatic empty-stage creation and enables `isaacmetalbridge.stage`, which opens the selected file through the real `omni.usd` context before `app ready`. The completion log includes the root layer and `/World` children so an ignored or overwritten stage cannot be mistaken for success. Without either scene option, the base experience opens an empty stage: Kit UI and input still work, but no environment asset is automatically referenced.

The familiar "grid" can mean either renderer-owned viewport guides on an empty stage or NVIDIA's `Simple Grid`/`default_environment.usd`, which is real USD environment geometry used by many Isaac Sim samples. The current diagnostic Metal hit/miss path does not reproduce the former overlay. The latter is fetched only when `--simple-grid` is explicitly requested. This is unrelated to M4 discovery: a valid run explicitly reports `Metal available=true device=Apple M4 unifiedMemory=true rayTracing=true`.

Dynamic `Create > Environments > Simple Grid` was reproduced against the real NVIDIA asset and enters Kit's sparse texture-streaming path. Sparse images are now enabled by default: Vulkan tile requirements and map/unmap operations are translated to real Metal sparse-heap mappings, and the real-container probe validates the round trip. Complete RTX texture/material equivalence still requires more image formats and descriptor/shader coverage.

The launcher can instead open the real NVIDIA asset as Kit's startup stage:

```sh
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --window --simple-grid --no-build
```

This explicit option fetches NVIDIA's real `default_environment.usd` and its three referenced wireframe PNGs into the ignored `build/runtime/assets` cache; it is not a bundled or mocked grid. The cache is mounted read-only and opened by the same early startup-stage extension, avoiding both the ignored positional-argument path and a live post-startup stage replacement. The verified Stage contains `Looks`, `GroundPlane`, `SphereLight`, and `Environment`; its geometry reaches Metal BLAS/TLAS and ray dispatch without a Render Graph failure. The Full viewer now presents a visible blue wire grid and axes for this stage. This is bridge scene presentation over real loaded geometry, not proof of general RTX material shading or camera/sensor equivalence. Use `--demo-scene` only for the smallest explicit regression scene and `--simple-grid` for the normal NVIDIA environment validation.

Expected non-fatal startup warnings include unavailable NVIDIA-only NVML, NGX, OptiX, Iray video decoding, and guest GLFW windowing. The guest intentionally runs with `--no-window`, and the native macOS viewer displays the bridged frame. Repeated `NVTT block-compression failed` messages are a remaining compatibility limitation that can affect compressed textures/materials; they are not treated as proof of a clean full RTX renderer.

## Repository Structure

```text
docs/                       Architecture, feasibility, integration, and roadmap
guest/                      C++20 guest probe and tests
host/                       Swift Metal host, Apple-container adapter, and tests
patches/apple-container/    Optional reviewable upstream patches
protocol/                   Versioned wire protocol
scripts/                    Build, test, doctor, setup, and patch helpers
third_party/apple-container Pinned upstream submodule (read-only)
```

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep experimental claims evidence-based, keep the upstream submodule clean, and add reviewable patch files instead of editing it directly.

## Security

Read [SECURITY.md](SECURITY.md). The prototype protocol has strict length and type checks but is not authenticated, encrypted, or ready for untrusted networks.

## License

A project license has not yet been selected. Until a license file is added by the repository owner, no license or redistribution permission is granted for IsaacMetalBridge source. The Apple submodule and all third-party components retain their own licenses.

## Trademark Notice

Apple, Metal, macOS, NVIDIA, CUDA, RTX, and Isaac Sim are trademarks of their respective owners. Use of a name is solely for technical identification and does not imply affiliation, cooperation, sponsorship, or endorsement.
