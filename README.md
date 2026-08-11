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
- Protocol 1.21 uploads bytes, runs real Metal compute, creates linear and sparse Metal textures, mirrors Vulkan sparse-image residency and ordered RGBA8 sRGB/BC3 sRGB/BC5 UNORM/RGBA32 float uploads into Metal, draws the tested raster/UI streams, creates translated Metal compute pipelines from Vulkan SPIR-V, transports buffer, image, texel-buffer, and independent sampler descriptors, builds Metal primitive/instance acceleration structures, and dispatches Metal rays with optional live Kit camera plus bounded positional USD light, DistantLight, and DomeLight arrays. SphereLight remains direct; RectLight and DiskLight carry transformed in-plane axes and half extents and use four deterministic surface samples with independent opacity-aware shadow rays and one-sided `-Z` emission. RectLight can sample its authored sRGB emission texture. CylinderLight carries its transformed local-X length axis and two radial extents, uses eight deterministic samples across its outward-emitting cylindrical surface, and reduces `treatAsLine` to two axial samples. Applied `UsdLuxShapingAPI` records add a transformed emission axis, cone angle, softness, focus, focusTint, and a bounded LM-63 Type C IES response, evaluated per emitter sample. It also defines a zero-AS Metal dispatch for the renderer-owned empty-stage guide grid.
- `DistantLight.inputs:angle` follows OpenUSD's angular-diameter definition over `[0, 360)`: zero keeps one parallel direction, while a nonzero angle uses four deterministic directions within the spherical cap and performs independent opacity-aware shadow traversal for each.
- The deterministic `[1,2,3,4] + 5 = [6,7,8,9]` path is verified both locally and across a real Apple-container Linux VM.
- A Linux ARM64 Vulkan loader discovers the experimental IMB device and completes one standard Vulkan compute flow through vsock and real Metal.
- A standard Vulkan image/render-pass/graphics-pipeline/`vkCmdDraw` flow produces a checked 64×64 RGBA8 triangle through vsock and real Metal; the PNG artifact is written under `build/artifacts`.
- The real Isaac Sim 6.0.1 ARM64 image loads the IMB Vulkan ICD and CUDA startup shim. Warp discovers `cuda:0` as the Apple-M4 compatibility device.
- The official `isaacsim.exp.base.kit` experience starts headlessly, loads Isaac core, PhysX, sensors, and `isaacsim.simulation_app`, then reports `app ready`.
- `--physics-smoke-output FILE` exercises the installed Isaac `DynamicCuboid`/`FixedCuboid`, real Kit timeline, and CPU PhysX scene. The verified Full run moved a 1 m cube from z=2.5 m to the correct resting center z=0.5 m in 73 updates and wrote checked JSON. CPU mode explicitly uses MBP broadphase, disables GPU dynamics, and enables transform readback; this does not claim GPU PhysX compatibility.
- The tested Kit UI vertex/index rings, font texture, scissor rectangles, and alpha-blended draw lists are bridged to Metal. A native macOS viewer shows the live real Isaac Sim menus, panels, toolbar, Stage, Property, and Content UI with `--window`.
- The native viewer forwards pointer, scroll, and keyboard events into the real Kit `carb.input.InputProvider`. Real Create/Window menu interaction, Stage search-field typing, Mesh-prim selection with the matching Property transform/material view, and opening/closing the Extensions Manager are verified end to end. At the correctness-first Warehouse rate of roughly 2 FPS, allow one or two rendered frames for a button release and the resulting panel update instead of treating the press frame as an unresponsive click.
- The KHR ray-tracing compatibility path records real scene BLAS geometry and builds Metal primitive acceleration structures. When Kit leaves its NVIDIA-only scene-instance descriptor null, atomic scene-state v19 also publishes each visible USD Mesh's real points, fan-triangulated face indices, authored corner normals, local bounds, world transform, material part, standard USD opacity/opacityThreshold, references into one globally deduplicated material texture table, and a separate bounded light texture table. Traversal includes native USD scenegraph instance proxies, and `materialBind` GeomSubsets are emitted as disjoint face sets instead of painting the whole source Mesh with one subset material. Bounded current-time `UsdGeomPointInstancer` expansion uses OpenUSD's own prototype-index, position, quaternion orientation, scale, prototype-root transform, instancer transform, IDs, and visibility-mask evaluation, then emits each surviving prototype Mesh under a synthetic stable instance path. The ICD builds one deterministic Metal BLAS per emitted Mesh occurrence and a Metal TLAS with the composed transforms; degenerate geometry normals and repeated/zero-area normal-map UV islands receive finite stable fallback bases instead of dropping the complete Mesh. v3-v18 inputs and v3/v4 bounds-based matching remain backward-compatible fallbacks. v19 preserves the separate camera and static Mesh/light/material sequences and appends one bounded complete light list, so camera-only navigation reads the 160-byte header plus its small light tail and reuses the existing TLAS instead of decoding and rematching the complete scene.
- The latest ray result is synchronized into Kit's actual triple-buffered 1280×720 viewport images before the final indexed UI pass. The included `--demo-scene` validation stage therefore displays a non-black real-geometry intersection image inside the live Kit application chrome.
- With `--window` and no explicit scene flag, Kit retains its normal `World`/`Environment` empty stage and no diagnostic USD is opened. Protocol 1.21 renders the familiar perspective grid and X/Y axes directly into the landscape viewport presentation textures, follows the published `/OmniverseKit_Persp` camera, and excludes Kit's separate font/icon atlas. A checked Full 1440×900 run loaded ROS 2 Jazzy, created and enabled all 204 Metal compute pipelines, reported zero Render Graph/multiTick/Sensor/Hydra/memory-map failures, and preserved readable Stage, Property, and Content UI.
- `--animate-demo` plays the explicit `--demo-scene` USD timeline. The stage extension samples the active timeline time for camera, transforms, Mesh topology/points/normals/UVs/display color, lights, native instance proxies, and PointInstancer data; the ICD rebuilds the fallback TLAS only when the atomic scene sequence changes and reuses content-identical Metal BLAS/material resources. A 300-frame Full run reached sequence 52, continued beyond 94 seconds, exited zero, and kept the Render Graph, multiTick, Sensor, Hydra, crash, and task-group failure counters at zero. The shipped stable demo animates transforms; arbitrary skinned or topology-changing deformation is not claimed.
- The launcher now waits for the real container init process and returns its status. A Kit crash after `app ready` is reported as a crash instead of being misreported as successful startup.
- Sparse buffer and sparse image residency are enabled by default. Vulkan sparse-image requirements, ordered buffer/image transfers, tile map/unmap operations, and mip storage are backed by Metal sparse textures; the Apple M4 probe validates RGBA8 sRGB map/unmap plus BC3 sRGB and BC5 UNORM upload/residency, while the real Isaac IES texture-cache path validates sparse RGBA32 float creation.
- A pinned Khronos SPIRV-Cross build translates live Isaac compute modules on the host. An ABI-preserving software-FP64 lowering retains each binary64 buffer slot while evaluating the validated scalar, vector, matrix, and `PackDouble2x32` operations through Metal float arithmetic. The host parses SPIR-V `DescriptorSet`/`Binding` plus `OpName` and correlates them with SPIRV-Cross's compact argument-buffer `[[id(N)]]` fields instead of incorrectly assuming that sparse Vulkan binding numbers are Metal IDs; resources in Vulkan sets 8 and above are emitted and bound as direct Metal buffer/texture/sampler arguments because Metal exposes only eight argument-buffer sets. Same-binding type aliases and SPIRV-Cross's generated numeric `m_ID` names are retained. Push constants are likewise matched from their SPIR-V `PushConstant` result ID/name to the exact translated Metal parameter, so an earlier direct set-8+ constant UBO cannot receive the push bytes. Protocol 1.20 preserves independent `VkSamplerCreateInfo` state, creates argument-buffer-capable `MTLSamplerState` objects, transports tightly packed ordinary 3D RGBA/BGRA images, and maps four-channel signed-normalized Vulkan images to Metal `rgba8Snorm`, and maps sparse `VK_FORMAT_R8G8B8A8_SRGB` images to Metal `rgba8Unorm_srgb` instead of rejecting Vulkan format 43. The current real-image trace creates and execution-enables all 204 requested `MTLComputePipelineState` objects. This includes 21 whose selected entry point only declares but never evaluates an FP64 matrix and fourteen exact true-use shaders whose captured SPIR-V dispatch/output is verified on Apple M4. The first thirteen fixtures cover decoded positions, sampled/storage 2D and 3D images, binary64 clipping paths, packed doubles, and 32-bit atomic counters. The final `f72cb1589be7be96` fixture checks its actual 64-bit hash-table compare-exchange path and reports `counter=1`, one occupied slot, and inserted key `0x17f0000010001`. Apple M4 cannot compile native MSL `atomic_ulong`; for this exact independent-global-ID kernel, the host preserves compare/update/returned-old-value semantics by running all logical invocations in one deterministic Metal thread. Kernels that use threadgroup/subgroup coordination are rejected from that lowering. The protocol-1.14 Full Warehouse/Franka rerun reached `app ready`, verified internal ROS 2 Jazzy, published 824 Meshes and 67 unique textures, built the 824-instance Metal TLAS, produced a visually checked textured 1440x900 frame, logged exactly 204 execution-enabled, 0 compile-only, and 0 skipped pipelines, reported zero targeted Render Graph/multiTick/Sensor/Hydra/format failures, and exited zero.
- A targeted ARM virtual-counter compatibility shim removes the backwards-TSC startup abort. The official Full experience reaches `Isaac Sim Full App is loaded`, reports `app ready`, displays the real Isaac UI, and exits zero in finite validation.
- `--simple-grid` opens NVIDIA's real `default_environment.usd` in either Base or Full. The current Full viewer shows its Stage prims plus a visible wire grid and axes. The host loads the asset's real 4096×4096 `Wireframe_blue.png` and samples it with the material's projected world-UV scale; this is the verified Simple Grid diffuse slice, not complete OmniPBR emission/mask equivalence.
- The startup-stage extension publishes the active Kit viewport camera position, basis, vertical field of view, and clip range through an atomic per-run state file. It also publishes up to eight supported positional USD lights (`SphereLight`, `RectLight`, `DiskLight`, or `CylinderLight`), four `DistantLight`s, and four `DomeLight`s, including position/direction and exposure-adjusted intensity. RectLight/DiskLight carry transformed local X/Y axes and half extents; RectLight additionally carries a bounded local sRGB `inputs:texture:file` image. CylinderLight carries its transformed local-X half length and local-Y/local-Z radial extents; Metal uses eight axial/radial samples with outward surface normals, or two axial samples when `treatAsLine` is true. Every area/line sample receives an independent opacity-aware shadow ray. Applied ShapingAPI lights also transport and evaluate cone, softness, focus, focusTint, `ies:angleScale`, `ies:normalize`, and a bounded local LM-63 Type C IES profile. The Vulkan ICD carries that state in protocol 1.21 ray submissions, and the Metal scene view applies the bounded light models instead of the former fixed camera and distance-only object shading.
- Scene-state v19 carries a direct bound `UsdPreviewSurface`/OmniPBR base color, standard `opacity` and `opacityThreshold`, `primvars:displayColor` fallback, or standard local-file `UsdUVTexture` images for base color, opacity, roughness, metallic, emission, and tangent-space normal. A separate opacity texture sharing the base UV source is composed into base alpha before transport; only explicitly tagged opacity samples affect visibility. Positive thresholds use bounded cutout re-intersection, while a zero threshold uses front-to-back source-over composition for up to 64 standard-opacity layers in linear color. SphereLight, every RectLight/DiskLight/CylinderLight sample, and DistantLight shadow rays use the same cutout decision and multiply fractional transmittance through up to 64 blockers. For the 12 MDL modules shipped by the official Simple Warehouse, it performs a bounded source-asset normalization without executing arbitrary MDL: channel-mask color blends, pushcart body/handle/cap masks, tint/desaturation, barcode/sign composition, ORM roughness min/max, and normal-strength reconstruction are baked exactly for those inputs. Base/emission textures are decoded from sRGB before lighting and the final linear result is encoded to sRGB; scalar, normal, and IES LUT maps remain linear. The official Barcode alpha and WallBoard `AlphaSelection` masks are baked into base-texture alpha; tagged hits below the authored `0.3333` threshold re-intersect the TLAS for up to 64 layers. The startup extension resolves standard connected UV networks and downsamples larger bounded scene images to at most 512px per axis for transport. Geometry retains a 192 MiB reservation, material images a separate 128 MiB reservation, light images a separate 16 MiB reservation, and Metal binds at most 126 unique material and light images. The verified Full Warehouse/Franka frame used 824 Mesh records including 12 tagged alpha-cutout records, 67 unique textures, 32.1 MiB geometry, and 77.8 MiB texture payload. Maps must share one complete UV source; scalar maps retain their selected R/G/B/A output and emission/normal use RGB. Colored absorption, refraction/volume/thin-walled semantics, authored tangent primvars, different UV sources within one material, UDIMs, mip generation, arbitrary MDL/MaterialX/RTX graphs, remaining Geometry/Portal light schemas, DomeLight HDR textures, LM-63 Type A/B or external TILT data, and full RTX lighting semantics remain unsupported and are tracked as correctness gaps rather than accepted final output.
- `--camera-sensor-output FILE` exports one RGB PPM plus JSON metadata from the Metal image driven by the active real Kit/USD camera. The default 640×480 output is resampled from the completed 1280×720 Metal camera frame. This avoids creating a second unsupported NVIDIA Render Graph, and a more-than-two-minute Full validation after `app ready` kept `Render Graph`, `multiTickRateRender`, `Sensor endFrame`, and `rtx.hydra Rendering failed` at zero. NVIDIA's downstream Replicator `rgb` array is still unsupported, so metadata records `dataSource: metal-camera-sensor` and `replicatorRgbDataReady: false`.
- This is not conformant Vulkan or CUDA. The visible scene path is still a bounded Metal raster/ray implementation, not full RTX/Hydra shader, material, lighting, nested-instancing, skinned/topology-deformation, camera-product, or sensor equivalence. Arbitrary graphics shaders, general CUDA kernels, validated GPU PhysX results, OptiX/NGX, native Replicator RGB/RTX sensor arrays, and RTX LiDAR remain unsupported.

- The 2026-08-11 protocol-1.20 Full validation used the real Isaac Sim 6.0.1 ARM64 image with bundled ROS 2 Jazzy, reported Apple M4 and 24,142 MB to Kit, execution-enabled all 204 captured compute pipelines, transported scene-state v18 with a complete eight-light list, and exited zero. The list included oriented RectLight, DiskLight, CylinderLight, a shaped SphereLight, two DistantLights, and two DomeLights; the host reported `lights=4/2/2`. The stage logged `/World/SpotLight` with axis `(0,0,-1)`, cone `35`, softness `0.35`, focus `4`, and focusTint `(0.12,0.28,1)`. The checked 1440x900 frame retains the Full workspace, textured sRGB geometry, transparent panel, shadows, and the SpotLight Stage entry. The former format-43 sparse-image rejection and all targeted Render Graph/multiTick/Sensor/Hydra errors were absent. NGX and OptiX remain NVIDIA-only and log three expected initialization errors.
- The 2026-08-11 protocol-1.21 Full validation transported scene-state v19 with an 8×8 RectLight emission image and a normalized Type C IES profile (`angleScale=1`, multiplier `12.290993`). Real Isaac's own IES processor requested sparse `VK_FORMAT_R32G32B32A32_SFLOAT`; the bridge created it through a Metal RGBA32-float sparse texture instead of returning `ERROR_FORMAT_NOT_SUPPORTED`. Isaac Sim Full 6.0.1, internal ROS 2 Jazzy, the eight-light list, the 9-Mesh fallback TLAS, and the native Full UI reached `app ready`, produced a checked 1440×900 frame, exited zero, and left no container behind. The earlier PPM/IES asset-parser crash and all targeted sparse/IES/scene-state errors were absent; only the three expected NVIDIA-only NGX/OptiX errors remained.

## Roadmap

1. Completed: validate Apple `container`, ARM64 Linux, and the real Isaac Sim compatibility failure.
2. Completed: connect a Linux guest to the macOS host through Apple's existing host-dialed vsock API without patching the submodule.
3. Completed at IMB-protocol level: real Metal buffer upload, compute, fence, and readback.
4. Completed prototype slice: Vulkan buffer/memory, one exactly recognized SPIR-V compute pipeline, command submission, fence, and readback map to verified IMB behavior.
5. Completed startup slice: inject the ICD and CUDA shim into the real Isaac Sim ARM64 image and reach `app ready` in the official base experience.
6. Completed interactive UI slice: real Isaac Kit UI draws through Metal in a native macOS window and receives pointer/keyboard input.
7. Completed sparse/compute/full-startup/scene slice: protocol 1.21 maps sparse Vulkan image tiles, including RGBA8 sRGB and RGBA32 float, to Metal, executes supported general compute with buffer/image/texel-buffer/sampler descriptors, builds Metal BLAS/TLAS resources, renders the no-USD empty-stage guide, and deterministically transports authored USD Mesh geometries plus the bounded material subset. Correctness-first output is the default: every output pixel receives its own primary ray and secondary opacity-aware shadow rays remain enabled at every scene size (`IMB_RAY_SHADOWS=always`, `IMB_RAY_PIXEL_STEP=1`). For explicit interactive performance experiments only, set `IMB_RAY_SHADOWS=adaptive IMB_RAY_PIXEL_STEP=adaptive`; that disables secondary shadows above 256 instances and uses 2x2/4x4 ray blocks above 256/512 instances. The Kit extension caches the large static scene payload until an OpenUSD objects-changed notice arrives, so camera and ordinary UI updates do not re-enumerate every Mesh and texture. Scene-state v19 preserves the separate camera/static sequences and adds global texture deduplication, GeomSubset material parts, bounded Warehouse Barcode/WallBoard alpha cutouts, standard USD opacity/opacityThreshold cutouts, linear source-over composition for fractional standard opacity, opacity-aware positional/DistantLight shadows, RectLight emission textures, and bounded Type C IES profiles; the ICD reads only the fixed header and reuses the existing TLAS for camera-only navigation. It also carries the active Kit camera plus bounded positional/Distant/Dome light arrays and exports one Metal-backed Camera Render Product. Next: broader material networks, colored transmission, Geometry/Portal light schemas, DomeLight HDR textures, broader IES variants, nested instancing and robust skinning/deformation, and native RTX sensor arrays.

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
- Active Kit viewport-camera position, basis, FOV, and clip-range transport into the Metal scene ray dispatch
- Bounded SphereLight, oriented four-sample RectLight/DiskLight, outward eight-sample CylinderLight (plus two-sample `treatAsLine`), applied ShapingAPI cone/softness/focus/focusTint, angular-diameter four-direction DistantLight, and DomeLight transport into Metal hit shading
- Atomic scene-state v19 transport of visible USD Mesh points/indices/authored corner normals/corner UVs, disjoint material-bound GeomSubset faces, deduplicated bounded RGBA8 material/light texture tables, direct roughness/metallic/emission constants, and a complete 16-record textured/shaped light list; deterministic per-Mesh Metal BLAS creation with derived tangent/bitangent data and authored world-transform TLAS instances; v3-v18 parsing and v3/v4 bounds matching remain available
- NVIDIA Simple Grid's real wireframe diffuse texture with its projected world-UV scale
- Connected `UsdUVTexture` fallback-color resolution plus standard local file-backed RGBA8 sampling through `UsdPrimvarReader_float2` and `UsdTransform2d`
- One-shot active-camera RGB output from a real Kit Camera Render Product via `--camera-sensor-output`
- Non-black 3D validation viewport for the included cube/ground stage and a visible wire-grid presentation for NVIDIA Simple Grid, composited into the real Kit UI
- Vulkan resource/synchronization compatibility needed for real Kit startup, including descriptor indexing, timeline semaphores, and query pools
- CUDA Driver startup identity, context, memory, stream, event, module, and private export-table compatibility sufficient for the tested extensions and Warp discovery
- Real Isaac Sim 6.0.1 official Base and Full experience startup to `app ready`
- Real Kit timeline plus CPU PhysX gravity/contact using Isaac's standard Cuboid APIs, with machine-readable output via `--physics-smoke-output`
- Current-time USD camera/light/Mesh/native-instance/PointInstancer sampling and stable looping transform animation via `--animate-demo`
- Apple-container patch apply/reverse workflow
- Non-downloading Isaac Sim environment checks

## Unsupported Features

- Conformant or general-purpose Vulkan rendering, arbitrary graphics shaders, full RTX/Hydra shader/material/light/camera equivalence, or general/native FP64 shader execution; only the validated software-lowered slice executes by default, and 64-bit atomic operations outside the validated serialized compare-exchange kernel remain unsupported
- General CUDA/PTX kernel execution or NVML
- Validated GPU PhysX results, Isaac Lab, native Replicator/RTX Camera annotator arrays, or RTX LiDAR output
- Nested PointInstancer expansion, per-instance primvar overrides, skinned/blend-shape or robust topology-changing deformation, arbitrary camera products, or WebRTC
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
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --robot-warehouse --no-build
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --no-build --quit-after 150
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid \
  --camera-sensor-output build/runtime/camera-rgb.ppm --no-build
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --demo-scene \
  --physics-smoke-output build/runtime/physics-smoke.json --no-build
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --demo-scene \
  --animate-demo --no-build
```

`--quit-after` is a frame count, not seconds. Success is the real Kit log line `app ready`, a zero container-init exit status, the launcher confirmation, and a window titled `Isaac Sim 6.0.1 — Apple Metal Bridge`. The launcher treats a nonzero exit after readiness as a crash. On a clean finite exit, the guest can close vsock between reading a final request and receiving its reply; the adapter accepts only that disconnect race and only after independently observing init status zero. The visible application chrome and its menu/search responses are generated by the real Isaac Kit process, not a mock UI. With `--simple-grid`, the current bridge should show the wire grid/axes rather than a fully black center; with `--robot-warehouse`, it should show NVIDIA's official Simple Warehouse plus Franka Panda; with `--demo-scene`, it should show the bounded orange Cube and gray Ground with geometry-derived lighting.

### Base, full, demo scene, and editor grid

The launcher defaults to the official `isaacsim.exp.base.kit` experience. `--experience full` selects `isaac-sim.sh`; a targeted virtual-counter shim handles the ARM VM timer path that previously aborted in `omni.anim.behavior.core`. Both profiles now reach `app ready`, and Full is the recommended profile when the complete Isaac application extension set is required.

`--demo-scene` selects [guest/scenes/metal-ray-scene.usda](guest/scenes/metal-ray-scene.usda). It is a local cube-and-ground validation scene, not NVIDIA's Simple Grid environment. Add `--animate-demo` to reset that stage to its start time, enable looping, and play its 0–48 time-code transform/PointInstancer animation; the flag is rejected without `--demo-scene` and is mutually exclusive with the separate `--physics-smoke-output` timeline test. Without it, the timeline remains stopped at Kit's current time. Isaac Sim 6.0.1's base experience ignores a positional USD argument because its legacy `/app/content/usdFile` opener is disabled. The launcher therefore disables automatic empty-stage creation and enables `isaacmetalbridge.stage`, which opens the selected file through the real `omni.usd` context before `app ready`. The completion log includes the root layer and `/World` children so an ignored or overwritten stage cannot be mistaken for success. Without `--demo-scene`, `--simple-grid`, or `--robot-warehouse`, the launcher opens the normal empty stage and does not insert a diagnostic USD.

The familiar "grid" can mean either renderer-owned viewport guides on an empty stage or NVIDIA's `Simple Grid`/`default_environment.usd`, which is real USD environment geometry used by many Isaac Sim samples. Protocol 1.21 reproduces the former as a Metal-generated guide in Kit's normal empty-stage viewport, following the live Kit camera without adding USD geometry. The latter is fetched only when `--simple-grid` is explicitly requested. This is unrelated to M4 discovery: a valid run explicitly reports `Metal available=true device=Apple M4 unifiedMemory=true rayTracing=true`.

Dynamic `Create > Environments > Simple Grid` was reproduced against the real NVIDIA asset and enters Kit's sparse texture-streaming path. Sparse images are enabled by default: Vulkan tile requirements, BC3 sRGB/BC5 UNORM buffer-to-image uploads, and map/unmap operations are translated to Metal sparse textures. The real-container probe validates both compressed uploads and the residency round trip on Apple M4. Complete RTX texture/material equivalence still requires more image formats and descriptor/shader coverage.

The launcher can instead open the real NVIDIA asset as Kit's startup stage:

```sh
ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --window --simple-grid --no-build
```

This explicit option fetches NVIDIA's real `default_environment.usd` and its three referenced wireframe PNGs into the ignored `build/runtime/assets` cache; it is not a bundled or mocked grid. The cache is mounted read-only and opened by the same early startup-stage extension, avoiding both the ignored positional-argument path and a live post-startup stage replacement. The verified Stage contains `Looks`, `GroundPlane`, `SphereLight`, and `Environment`. Scene-state v19 carries its visible two-triangle Mesh geometry, authored normals when present, and USD world transform into a dedicated Metal BLAS. The host samples the real `Wireframe_blue.png`, and the Full viewer presents the resulting wire grid and axes. This is a verified slice of the actual asset, not proof of general OmniPBR or native sensor equivalence. Use `--demo-scene` only for the smallest explicit regression scene and `--simple-grid` for the normal NVIDIA environment validation.

`--camera-sensor-output` requires one of those explicit stages. It creates a 640×480 RGB sensor file by default; set `IMB_CAMERA_SENSOR_WIDTH` and `IMB_CAMERA_SENSOR_HEIGHT` to values from 16 through 8192 before launch to change it. The PPM and `<file>.json` are copied atomically to the requested path as soon as capture completes, so they remain available if an interactive run is later stopped with Ctrl-C. Metadata includes the USD camera path, dimensions, byte type, data source, CRC32, and whether NVIDIA's native Replicator `rgb` CPU array became ready. The current validated source is `metal-camera-sensor`: it is the active real Kit camera's bounded Metal frame contract, not native RTX Camera annotator equivalence.

`--physics-smoke-output` requires the explicit local `--demo-scene`. It adds Isaac's standard `DynamicCuboid` and `FixedCuboid` only to the in-memory session layer, configures a CPU PhysX scene with MBP broadphase, starts the real Kit timeline, and pauses after contact. The JSON records initial/final positions, drop distance, timeline state, update count, and pass/fail. The requested host file is copied atomically while an interactive viewer remains open. This validates CPU rigid-body gravity and contact; GPU PhysX/CUDA kernel correctness remains unsupported.

Expected non-fatal startup warnings include unavailable NVIDIA-only NVML, NGX, OptiX, Iray video decoding, and guest GLFW windowing. Isaac Sim 6.0.1 Full also emits `Python import process in omni.anim.graph.core failed`; an isolated run without the bridge extensions, startup stage, or host adapter reproduces the identical warning in NVIDIA's bundled deprecated Animation Graph extension, while Full continues loading. The guest intentionally runs with `--no-window`, and the native macOS viewer displays the bridged frame. Full always enables and validates the bundled ROS 2 Jazzy core, internal `rclpy`, and bridge; `--ros2` is retained as a Full-only compatibility no-op. RGBA8 sRGB, BC3 sRGB, and BC5 UNORM sparse residency now reaches Metal, but other NVTT or RTX material warnings are not treated as proof of full RTX equivalence.

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
