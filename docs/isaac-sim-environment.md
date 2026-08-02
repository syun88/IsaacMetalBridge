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
5. **Feature startup:** Warp discovers the compatibility CUDA device and supported Vulkan compute/ray work reaches Metal. The active Kit camera, first USD SphereLight/DistantLight/DomeLight, visible Mesh points/indices/transforms/normals/UVs, bounded local-file base-color/roughness/metallic/emission/tangent-normal textures, direct material constants, real Simple Grid diffuse texture, and one Camera Render Product file contract now reach the bounded Metal path. General CUDA/PTX, GPU PhysX correctness, broader material networks, native RTX sensor arrays, OptiX, and NGX require deeper implementation.

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

On 2026-07-30, Phases 0 through 4, the interactive Kit UI, sparse-image/compute probes, and the bounded Metal ray-validation milestone passed with Apple `container` 1.1.0. The official `nvcr.io/nvidia/isaac-sim:6.0.1` ARM64 image is the base of the derived image; its Omniverse Kit, Carbonite, PhysX, USD, Python, and Isaac extensions remain the real NVIDIA distribution. The IMB Vulkan ICD passes real loader/Metal storage-buffer and texel-buffer compute, transfer/image readback, sparse-image map/unmap, raster, KHR BLAS/TLAS, OPAQUE_FD, and ray-dispatch tests. It supplies the resource, descriptor, synchronization, timeline-semaphore, query-pool, and tested UI-raster subset needed during startup. The CUDA shim supplies the tested Driver API, virtual-memory, and private export-table startup surface. Warp reports CUDA Toolkit 12.9, Driver 12.8, and `cuda:0` as `IsaacMetalBridge CUDA-compat (Apple M4)`.

The official `isaacsim.exp.base.kit` and Full `isaac-sim.sh` experiences load the real Isaac extension stack and report `app ready`. A targeted ARM virtual-counter shim handles the backwards-TSC path that previously aborted Full startup. Observed Kit UI vertex/index rings, texture, scissors, and indexed draws are executed by Metal and displayed in a native macOS window. Real scene BLAS geometry is built into Metal acceleration structures, intersected by Metal, and synchronized into Kit's viewport presentation ring. Isaac Sim base does not open a positional USD in this release, so `isaacmetalbridge.stage` opens only an explicitly selected stage through `omni.usd` during startup and logs its root and `/World` children. Without `--demo-scene` or `--simple-grid`, the launcher does not insert a diagnostic USD. The viewer returns mouse and keyboard records to a separate Kit extension that injects them through Carbonite's input-provider API. Use `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid`; the script never supplies EULA acceptance on the user's behalf.

For a live bounded timeline check, use `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --demo-scene --animate-demo --no-build`. `--animate-demo` is valid only with the explicit local demo stage. It resets the real Kit timeline to the stage start, enables looping, and plays the 0–48 time-code interval at 24 time codes per second. The extension converts timeline seconds through the stage rate and samples camera, transforms, Mesh topology/points/normals/UVs/display color, lights, native instance proxies, and PointInstancer attributes at that time. It publishes only changed atomic payloads. The ICD then refreshes the fallback TLAS for a new sequence and reuses content-identical BLAS/material resources for transform-only motion. The stable shipped scene animates Cube/instance transforms rather than topology. A 300-frame Full validation reached sequence 52, passed 94 seconds, exited zero, and kept Render Graph, multiTick, Sensor, Hydra, crash, and task-group failure patterns at zero.

The boundary remains substantial: the non-black scene viewport uses a bridge-controlled fallback TLAS and bounded shading path. Scene-state v11 atomically publishes the active Kit camera, first USD SphereLight/DistantLight/DomeLight, and each visible Mesh's real points, triangulated indices, local bounds, world transform, resolved material mode, and normalized authored corner normals. It resolves `constant`, `uniform`, `vertex`, `varying`, and `faceVarying` USD normal and UV primvar interpolation before transport. Material base color supports direct Preview/OmniPBR constants, display color, a connected no-file `UsdUVTexture` fallback, or a standard connected local image through `UsdPrimvarReader_float2`; a connected `UsdTransform2d` chain is evaluated in scale-rotate-translate order and baked into the corner UVs. Images are bounded to 4096x4096 and transported as RGBA8 for repeat-wrapped linear Metal sampling. Roughness, metallic, RGB emission, and RGB tangent-space normal can also come from connected local images sharing the same complete UV source. For normal images the ICD derives tangent/bitangent vectors from real triangle positions and transformed UV derivatives; Metal transforms and orthogonalizes the TBN basis against the interpolated normal. A 2026-08-02 Full demo built Cube and Ground into one TLAS, reported `normals=authored-corner`, published `parameterTextures=2 normalTextures=1 uvTransforms=1`, logged Cube `normal:8x8[rgb-tangent]`, and visibly rendered transformed normal-dependent checker lighting plus the cast shadow. The same run reached `app ready`, exited zero, and kept the four targeted Render Graph/multiTick/Sensor/Hydra failure patterns at zero. NVIDIA's native RTX texture loader still reported `ERROR_FORMAT_NOT_SUPPORTED`; the successful result came from the explicit scene-state-v11 Metal path. The host separately samples the real 4096x4096 `Wireframe_blue.png` using its authored projected world-UV scale. A verified Simple Grid Full run reported `/World` children `Looks`, `GroundPlane`, `SphereLight`, and `Environment`, presented the wire grid/axes, reached `app ready`, and exited zero.

`--camera-sensor-output` additionally exports the active real Kit/USD camera's completed Metal frame. The verified output resampled 1280×720 to 640×480 and produced 921,600 checked RGB bytes plus JSON metadata. The metadata truthfully records `dataSource: metal-camera-sensor` and `replicatorRgbDataReady: false`: NVIDIA's native downstream `rgb` CPU array still depends on unsupported RTX post-processing compute, so this is a bounded bridge contract rather than native RTX Camera equivalence. Avoiding an extra Render Product also removed the repeated Render Graph, multiTick, Sensor endFrame, and Hydra failures in a more-than-two-minute Full run after `app ready`.

`--physics-smoke-output` validates a separate real CPU PhysX slice. It authors Isaac's installed `DynamicCuboid` and `FixedCuboid` into the explicit demo-stage session, sets MBP broadphase, disables GPU dynamics, enables transform readback, plays the real Kit timeline, and pauses after contact. The verified Full run advanced 73 updates and moved the cube from z=2.5 m to z=0.49999994 m, with a 2.00000006 m drop recorded as `passed: true`. The same run kept the four targeted Render Graph/multiTick/Sensor/Hydra error patterns at zero. This is not evidence for GPU PhysX, articulations, or arbitrary collision workloads.

General material networks, authored tangent/normal-strength semantics, nested instancing and per-instance primvar overrides, animated material inputs, robust skinning/blend-shape/topology deformation, remaining light schemas and soft/area-light shadows, native RTX sensor arrays, NVML, and `nvidia-smi` are absent; CUDA/PTX and GPU PhysX results are not generally implemented. Native USD scenegraph instance proxies, bounded current-time PointInstancer expansion, authored interpolated normals (with geometric-face fallback), derived tangent-space normal mapping, and SphereLight/DistantLight hard-shadow rays do reach Metal. A Full demo expanded two references plus three visible of four PointInstancer entries from one abstract pyramid prototype, reported `instanceProxyMeshes=2 pointInstances=3 pointInstanceMeshes=3`, and visibly rendered all five occurrences with their different transforms. Sparse image residency is enabled by default: Vulkan tile requirements and map/unmap calls reach real Metal sparse heaps and pass the container probe. `--simple-grid` caches the real root USDC and its three wireframe PNG dependencies under ignored `build/runtime`, mounts them read-only, and opens the root through the startup extension. That extension now starts after `omni.kit.mainwindow`, preserving the genuine docked Full Viewport/Stage/Property/Content workspace. NVIDIA-only NGX/OptiX remain unavailable. The stable Full path clears both the global ROS environment and Isaac's default ROS bridge setting unless `--ros2` is explicitly selected; `--ros2` remains experimental. The verified result must be described as real Full Isaac startup and bounded Metal execution, not full RTX or general CUDA compatibility.
