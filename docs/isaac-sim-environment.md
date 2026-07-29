# Real Isaac Sim environment

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Validation baseline

The required research baseline is **NVIDIA Isaac Sim 6.0.1 for Linux aarch64**. NVIDIA publishes both a Linux aarch64 standalone package and the multi-architecture `nvcr.io/nvidia/isaac-sim:6.0.1` container. This is the real Isaac Sim distribution; the project does not create a substitute environment.

Primary NVIDIA references:

- [Download Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
- [Quick Install](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html)
- [Container Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Isaac Sim Requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html)

## Availability and support boundary

Linux aarch64 availability is verified in NVIDIA's documentation. Apple Silicon compatibility is **not**. NVIDIA's documented aarch64 requirement for Isaac Sim 6.0 is DGX Spark, DGX OS 7, and an NVIDIA driver. NVIDIA also requires an RTX-capable NVIDIA GPU for the compatibility check. Apple GPU/Metal therefore cannot satisfy the published GPU contract without a new compatibility layer.

## Installation methods

1. **External standalone directory:** extract the official Linux aarch64 archive outside this repository, run NVIDIA's `post_install.sh` in its Linux guest, and point `ISAAC_SIM_PATH` to it.
2. **External OCI image:** manage `nvcr.io/nvidia/isaac-sim:6.0.1` with the user's container/image store and set `IMB_ISAAC_IMAGE`. Do not export or copy the image into this Git tree.

The setup script validates configuration only. It does not download, extract, authenticate to NGC, or accept licenses.

## Container/image requirements

- Linux `arm64`/`aarch64` manifest and user space
- `/isaac-sim` layout for the official container
- user-provided `ACCEPT_EULA=Y` only after the user has reviewed and accepted NVIDIA's terms
- writable external cache, data, config, and log locations
- outbound HTTPS or separately managed local asset packs
- the IMB Vulkan ICD and CUDA startup shim for the currently implemented compatibility subset

## Dependencies that must be preserved

- Omniverse Kit and Carbonite plugins
- PhysX and USD runtime
- Isaac Sim Python 3.12 environment and extensions
- glibc and Linux aarch64 dynamic loader expectations
- the exact shared libraries shipped by NVIDIA

## Expected first failure points

1. **Guest startup:** verified for a minimal ARM64 guest and for a shell override in the real Isaac Sim image.
2. **Dynamic loading:** the IMB-derived image now supplies the Vulkan ICD and CUDA startup library; NVML and `nvidia-smi` remain unavailable.
3. **Compatibility checker:** it reaches `app ready`, but still reports an overall failure for the missing NVIDIA management stack. On the tested 32 GiB Mac, the launcher now assigns 24 GiB to the guest by default and Warp reports a 24 GiB compatibility device.
4. **Kit startup:** the official Base and Full experiences reach `app ready`; the tested UI draw stream and bounded scene presentation reach Metal.
5. **Feature startup:** Warp discovers the compatibility CUDA device and supported Vulkan compute/ray work reaches Metal, but general CUDA/PTX, GPU PhysX correctness, RTX materials/cameras/sensors, OptiX, and NGX require deeper implementation.

## Milestone procedure

```text
Phase 0  container CLI/system verified
Phase 1  uname -m == aarch64 in a real Linux guest
Phase 2  official compatibility checker or Isaac Sim launcher genuinely executed
Phase 3  first missing API/library/device call captured from real logs/traces
Phase 4  official base experience reaches app ready through IMB
```

Record exact command, image/package digest, exit status, and unedited log location outside Git. A plausible launch narrative is not evidence.

## Current result

On 2026-07-29, Phases 0 through 4, the interactive Kit UI, sparse-image/compute probes, and the bounded Metal ray-validation milestone passed with Apple `container` 1.1.0. The official `nvcr.io/nvidia/isaac-sim:6.0.1` ARM64 image is the base of the derived image; its Omniverse Kit, Carbonite, PhysX, USD, Python, and Isaac extensions remain the real NVIDIA distribution. The IMB Vulkan ICD passes real loader/Metal storage-buffer and texel-buffer compute, transfer/image readback, sparse-image map/unmap, raster, KHR BLAS/TLAS, OPAQUE_FD, and ray-dispatch tests. It supplies the resource, descriptor, synchronization, timeline-semaphore, query-pool, and tested UI-raster subset needed during startup. The CUDA shim supplies the tested Driver API, virtual-memory, and private export-table startup surface. Warp reports CUDA Toolkit 12.9, Driver 12.8, and `cuda:0` as `IsaacMetalBridge CUDA-compat (Apple M4)`.

The official `isaacsim.exp.base.kit` and Full `isaac-sim.sh` experiences load the real Isaac extension stack and report `app ready`. A targeted ARM virtual-counter shim handles the backwards-TSC path that previously aborted Full startup. Observed Kit UI vertex/index rings, texture, scissors, and indexed draws are executed by Metal and displayed in a native macOS window. Real scene BLAS geometry is built into Metal acceleration structures, intersected by Metal, and synchronized into Kit's viewport presentation ring. Isaac Sim base does not open a positional USD in this release, so `isaacmetalbridge.stage` opens only an explicitly selected stage through `omni.usd` during startup and logs its root and `/World` children. Without `--demo-scene` or `--simple-grid`, the launcher does not insert a diagnostic USD. The viewer returns mouse and keyboard records to a separate Kit extension that injects them through Carbonite's input-provider API. Use `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid`; the script never supplies EULA acceptance on the user's behalf.

The boundary remains substantial: the non-black scene viewport uses a bridge-controlled fallback TLAS and scene-presentation camera/shading path. General scene transforms, live Kit camera state, translated RTX/Hydra shaders, complete materials/lighting, RTX sensor output, NVML, and `nvidia-smi` are absent; CUDA/PTX and GPU PhysX results are not generally implemented. Sparse image residency is enabled by default: Vulkan tile requirements and map/unmap calls reach real Metal sparse heaps and pass the container probe. `--simple-grid` caches the real root USDC and its three wireframe PNG dependencies under ignored `build/runtime`, mounts them read-only, and opens the root through the early startup extension. A verified Full run reported `/World` children `Looks`, `GroundPlane`, `SphereLight`, and `Environment`, presented a visible blue wire grid/axes in the native viewer, reached `app ready`, and exited zero. This presentation does not establish full RTX material equivalence. NVIDIA-only NGX/OptiX remain unavailable. The stable Full path intentionally clears the global ROS environment; its ROS 2 extension logs a non-fatal startup failure, while `--ros2` remains experimental. The verified result must be described as real Full Isaac startup and bounded Metal execution, not full RTX or general CUDA compatibility.
