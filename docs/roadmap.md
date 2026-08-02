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
- protocol 1.13 session verified across a real Apple-container ARM64 VM
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
- completed: pinned SPIRV-Cross tool and protocol 1.13 live compute-pipeline creation; 162 of 204 real Isaac requests compile on M4
- completed: generic Metal compute dispatch with push constants, buffer/image descriptors, and Vulkan-format texel-buffer views; the Full trace executes supported pipelines and safely skips unsupported/partially-bound RTX submissions
- completed: KHR triangle BLAS and instance TLAS compatibility objects backed by real Metal acceleration structures
- completed: real Metal ray dispatch against real Kit scene BLAS geometry and synchronization into Kit's triple-buffered 1280×720 viewport images
- completed: non-black `--demo-scene` viewport inside the live Kit UI, with mouse menu interaction and Stage-search keyboard input verified end to end
- completed: early `omni.usd` startup-stage extension replaces Isaac Sim base's disabled positional-USD path; local demo and real NVIDIA `Simple Grid` roots are verified by their `/World` children
- completed: scene-state version 3 publishes visible USD Mesh triangle counts, local bounds, material flags, and world transforms; the ICD matches these against computed real BLAS bounds, applies the authored transforms, and excludes renderer-internal BLASes from the null-descriptor fallback TLAS
- completed: scene-state version 9 publishes real Mesh points, triangulated face indices, normalized authored corner normals, triangulated corner UVs, bounded RGBA8 base-color pixels, and direct roughness/metallic/emission constants; the ICD deterministically builds one Metal BLAS per authored Mesh, with Cube and Ground verified together in a Full TLAS
- completed: direct bound `UsdPreviewSurface`/OmniPBR base color plus `primvars:displayColor` fallback transported per Mesh and consumed through Metal instance user IDs
- completed: follow connected `UsdUVTexture.outputs:rgb/rgba`; resolve the standard no-file fallback or decode a local file, resolve `UsdPrimvarReader_float2`, transport all five primvar interpolation modes, and barycentrically sample repeat-wrapped UVs in Metal; Full rendered the 8x8 checker PNG on Cube
- completed: Vulkan sparse-image memory requirements, mip storage, and tile map/unmap mirrored into real Metal sparse heaps and enabled by default
- completed: real R32_UINT texel-buffer dispatch plus buffer/image descriptor and readback transport
- completed: active Kit viewport camera plus the first USD SphereLight, DistantLight, and DomeLight published atomically by scene-state v4-v7 and used by protocol 1.13 Metal ray dispatch
- completed: NVIDIA Simple Grid's real `Wireframe_blue.png` sampled using its authored projected world-UV scale
- completed: active Kit viewport-camera scene trace into the 1280×720 Metal presentation image
- completed: `--camera-sensor-output` one-shot RGB/JSON contract resampled from that completed active-camera Metal frame; verified 921,600 RGB bytes and checksum in Full without a second failing Render Graph
- completed: real Kit timeline and CPU PhysX rigid-body gravity/contact slice using Isaac `DynamicCuboid`/`FixedCuboid`; Full reached the correct z=0.5 m resting center with MBP broadphase and emitted checked JSON
- next: lower the 42 FP64-buffer compute requests while preserving Vulkan ABI and expand unsupported descriptor/format coverage
- completed: traverse native USD scenegraph instance proxies and emit each composed proxy path as a separate scene-state-v11 Mesh/world-transform record; a Full demo verifies two references to one abstract pyramid prototype as `instanceProxyMeshes=2`, four Metal BLASes, and one four-instance TLAS
- completed: expand bounded current-time `UsdGeomPointInstancer` prototypes through OpenUSD's own prototype-index, position, quaternion orientation, scale, prototype-root/instancer transform, ID, inactive/invisible mask semantics; a Full demo authored four points, masked one, reported `pointInstances=3 pointInstanceMeshes=3`, and built the expected seven-Mesh TLAS
- completed: convert `omni.timeline` seconds through the stage time-code rate and sample camera, transforms, Mesh topology/points/normals/UVs/display color, lights, native instance proxies, and PointInstancer attributes at that time; `--animate-demo` loops the explicit demo stage, publishes only changed atomic payloads, and refreshes the fallback TLAS while content-identical Metal BLAS/material resources are reused. A 300-frame Full run reached scene sequence 52, passed 94 seconds, exited zero, and reported none of the targeted Render Graph/multiTick/Sensor/Hydra/crash/task-group failures
- next: preserve nested instancers and per-instance primvar overrides, translate animated material inputs, and make skinning/blend-shape/topology-changing deformation robust in Full
- completed: preserve per-triangle geometric face normals at BLAS build time, transform them into world space for each TLAS instance, and use them for bounded SphereLight/DistantLight diffuse response; a rotated-instance Metal regression proves lighting differs from the former view-facing fallback
- completed: resolve USD Mesh `constant`, `uniform`, `vertex`, `varying`, and `faceVarying` authored normals to triangulated corners in scene-state v6+, transform them per instance, and barycentrically interpolate them in the Metal ray hit shader; a planar gradient regression verifies the interleaved position/normal path
- completed: trace a second ray through the real TLAS for SphereLight and DistantLight visibility; an occluder regression proves the primary hit stays fixed while direct illumination drops, and the Full demo visibly casts the Cube shadow onto Ground
- completed: direct `UsdPreviewSurface`/OmniPBR roughness and metallic constants reach a per-instance Metal material buffer and bounded dielectric-to-metal diffuse/specular response; a real-ray regression distinguishes rough dielectric from smooth metal and Full logs verify Cube 0.32 versus Ground 0.82 roughness
- completed: direct Preview/OmniPBR emissive color/intensity reaches the per-instance Metal material record and is added without a light; a no-light regression verifies green self-emission and Full v9 logs Cube `(0.060,0.015,0.000)x1.000`
- completed: scene-state v10 resolves file-backed roughness/metallic/emission connections sharing one UV primvar, retains scalar channel selectors, creates separate Metal images plus a validated per-material descriptor, and a spatial emission-map regression rejects average-color substitution; Full logs two 8x8 Cube parameter maps
- completed: scene-state v11 resolves one connected tangent-space normal image, derives tangent/bitangent vectors from real triangle positions and UV derivatives, transports vertex format 6, and orthogonalizes the transformed TBN basis in Metal; a directional-light regression and Full `normalTextures=1` run verify spatial normal response
- completed: resolve a standard connected `UsdTransform2d` chain, apply its authored scale then counter-clockwise degree rotation then translation to triangulated corner UVs, require every map on the bounded material to share that complete UV source, and verify the non-identity demo network in real Full with `uvTransforms=1`
- next: translate broader hit/miss/material networks, authored tangent/normal-strength semantics, soft/area-light shadows and remaining light schemas, broader raster commands, and NVIDIA's native Replicator/RTX sensor arrays

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
- stable Full startup isolates the bundled ROS environment and clears the default bridge setting unless requested; explicit `--ros2` remains experimental

GPU PhysX correctness, broader CPU PhysX workloads, general RTX/Hydra shader/material/camera-sensor equivalence, long-running Full workload stability, and general CUDA compatibility remain research questions, not scheduled promises. The completed scene-validation milestone displays multiple authored visible USD Meshes, native scenegraph instance proxies, and bounded current-time PointInstancer expansions through Metal with their composed world transforms, authored interpolated or geometric fallback normals, tangent-space normal maps, transformed UVs, SphereLight/DistantLight hard shadows, active Kit camera, first SphereLight/DistantLight/DomeLight, real Simple Grid diffuse texture, and one Camera Render Product file contract. It still uses a fallback TLAS and simplified bounded light response; nested instancing, per-instance primvar overrides, animated material inputs, robust skinning/deformation, soft/area-light shadows, and NVIDIA's native RGB annotator or other RTX sensors are not equivalent.
