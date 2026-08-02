# Minimal Vulkan ICD

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Verified scope

The Linux ARM64 guest image contains an experimental Vulkan Installable Client Driver at `/usr/local/lib/libimb_vulkan_icd.so` and a loader manifest at `/usr/local/share/vulkan/icd.d/imb_icd.json`.

The implementation follows Khronos's documented loader/driver discovery ABI:

- exports `vk_icdNegotiateLoaderICDInterfaceVersion`
- exports `vk_icdGetInstanceProcAddr`
- negotiates loader/driver interface version 5 or lower
- uses a version 1.0.1 JSON manifest selected with `VK_DRIVER_FILES`
- initializes dispatchable handles with the loader magic from `vk_icd.h`

Primary reference: [Khronos Vulkan Loader driver interface](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md).

The current driver combines exact Metal compute, fixed raster, and tested Kit UI raster paths with a broader startup compatibility surface:

- Vulkan 1.1 instance/device identity and creation/destruction
- one physical device whose name comes from the live IMB/Metal capability reply
- the feature/limit and extension subset requested by the tested Kit startup, including descriptor indexing, memory budget, dedicated allocation, and timeline semaphores
- memory and queue-family property reporting for the existing shared-buffer/compute research path
- one logical device and 16 queues
- device/queue idle operations
- storage-buffer creation, host-visible/coherent memory, binding, mapping, and readback
- descriptor layouts/sets for storage and uniform buffers, sampled/storage images, and uniform/storage texel buffers
- exact `ADD_U32` plus live SPIR-V-to-MSL compute pipelines
- primary command-pool/buffer recording for bind, push constants, and three-dimensional dispatch
- `vkQueueSubmit`, Vulkan fence wait/status/reset, real Metal fence completion, and result readback
- opaque startup resources for images/views/samplers, render passes/framebuffers, graphics pipelines, pipeline caches, and general descriptor layouts
- one executable RGBA8 image/render-pass/framebuffer/fixed-triangle graphics pipeline with `vkCmdDraw`, queue submit, Metal fence, and checked readback
- one exactly recognized Kit UI graphics pipeline with BGRA8 targets, texture descriptors, copy/upload tracking, indexed draws, ring-buffer offsets, scissors, and synchronous Metal completion
- KHR ray-tracing pipeline/group/SBT compatibility sufficient for the verified probe and observed real-Isaac trace
- KHR triangle BLAS and instance TLAS builds backed by real Metal acceleration structures
- bounded real Metal ray dispatch into RGBA8 for the active Kit viewport camera
- synchronization of the 1280×720 real-Isaac result into Kit's triple-buffered viewport images and one-shot RGB sensor publication at the requested output size
- Vulkan sparse-image requirements and tile map/unmap backed by Metal sparse heaps
- real generic compute execution for supported translated pipelines with buffer, image, and formatted texel-buffer bindings
- binary/timeline semaphores and query-pool lifecycle/results
- synchronous validation/no-op handling for generic graphics submissions outside the fixed triangle and recognized Kit UI paths

The ICD accepts a host connection only when `IMB_VSOCK_PORT` is explicitly set. It performs IMB 1.13 negotiation and requires the live host to advertise Metal buffer, compute, image, fixed raster, UI raster, resource-I/O, and real-fence capabilities before exposing the physical device. Live SPIR-V compute-pipeline creation requires `METAL_SPIRV_COMPUTE`; sparse images require `METAL_SPARSE_IMAGE`; acceleration-structure and ray commands require `METAL_ACCELERATION_STRUCTURE` and `METAL_RAY_DISPATCH`.

## Explicit limitations

This is not a conformant Vulkan implementation and does not report a conformance version. Compute shader modules are sent through the protocol to a pinned SPIRV-Cross build and compiled as real Metal pipelines when supported; the current live Isaac trace creates 162 of 204 requested pipelines and rejects 42 FP64-buffer modules explicitly. Supported generic dispatches execute on Metal. Legal null/partially-bound descriptors and unsupported formats are skipped without failing the entire RTX queue; a measured finite Full run executed 595 Metal dispatches and skipped 2,708 such submissions. General graphics pipelines remain compatibility objects/no-ops outside the verified fixed and Kit UI paths.

Real Isaac RTX creates a small R32_UINT view whose explicit byte range includes one incomplete final texel. The compatibility ICD accepts only that bounded tail case by reducing 1161 requested bytes to 1160 usable bytes. The real-container probe covers both the standard `VK_WHOLE_SIZE` form and this explicit-tail compatibility form; other zero, out-of-buffer, misaligned-offset, and oversized views still fail.

The verified ray path intersects real scene geometry, but Kit's NVIDIA-only instance producer leaves the observed TLAS descriptor null. Scene-state version 11 preserves the earlier camera/light/geometry/normal/base-texture/material/emission/parameter-map fields, then adds one tangent-space normal texture record. The Kit extension resolves `constant`, `uniform`, `vertex`, `varying`, and `faceVarying` USD Mesh normal and float2 primvar interpolation. It enables native USD scenegraph instance-proxy traversal and bounded current-time PointInstancer expansion. `omni.timeline` seconds are converted with the stage time-code rate before OpenUSD evaluates camera, transforms, Mesh topology/points/normals/UVs/display color, lights, proxy transforms, prototype indices, positions, quaternion orientations, scales, prototype-root and instancer transforms, IDs, inactive IDs, and invisible IDs. Every surviving prototype Mesh is emitted under a synthetic stable path and composed world transform. The extension publishes only when the atomic payload changes. The ICD keys its fallback TLAS by sequence and content-hashes geometry/material data independently of path and world transform, so transform-only occurrences share Metal BLAS/material resources. The extension also follows connected NodeGraph outputs to either the no-file `UsdUVTexture` fallback-color case or standard local files plus `UsdPrimvarReader_float2`; standard connected `UsdTransform2d` chains are evaluated as scale then counter-clockwise degree rotation then translation and baked into the corner UVs. Decoded textures are capped at 4096x4096. Maps must share one complete UV source in v11, scalar R/G/B/A output selection is retained, and emission/normal use RGB. The ICD builds a dedicated Metal BLAS for every unique transported Mesh content record and applies each occurrence's USD world transform in the fallback TLAS. Material geometry duplicates indexed corners into interleaved position/normal/UV/material/emission data; vertex format 6 adds normalized tangent and bitangent vectors derived from actual triangle positions and transformed UV derivatives. Separate host RGBA8 images are referenced through a validated 48-byte `MBM1 v1` or 56-byte `MBM1 v2` descriptor. Metal binds up to 16 unique images, transforms and orthogonalizes the TBN basis against the interpolated authored normal, samples repeat-wrapped tangent-space normals, applies bounded dielectric-to-metal diffuse/specular response, and adds mapped self-emission without a light. Scene-state v3-v10 remains accepted. A 2026-08-02 Full run deterministically built Cube, Ground, two native references, and three visible of four PointInstancer entries from one pyramid prototype in one TLAS. It published `parameterTextures=2 normalTextures=1 uvTransforms=1 instanceProxyMeshes=2 pointInstances=3 pointInstanceMeshes=3`, logged seven scene-state-v11 Mesh occurrences, visibly rendered all five prototype occurrences plus the transformed normal-mapped Cube, and exited zero. A later 300-frame transform-animation run reached sequence 52, passed 94 seconds, and also exited zero without the targeted Render Graph/multiTick/Sensor/Hydra/crash/task-group failures. NVIDIA's own sparse RTX texture request remained unsupported; this bounded v11 path is independent of general RTX texture semantics. Authored tangent primvars, normal scale/strength, different UV sources within one material, UDIMs, mip generation, exact color management, nested instancing and per-instance primvar overrides, animated material inputs, robust skinning/deformation, and general MaterialX/OmniPBR graphs remain incomplete.

For ray submissions, the ICD also sends the active Kit viewport camera plus the first USD SphereLight, DistantLight, and DomeLight to Metal. Directional world transform, color, exposure-adjusted intensity, radius/angle, and dome ambient color are validated before use. The host retains either one normalized geometric face normal or three authored corner normals for every triangle in a supported single-geometry BLAS before Vulkan source buffers can disappear. It applies the TLAS instance cofactor transform, indexes the resulting world normals by Metal instance/primitive ID, and barycentrically interpolates authored triples at the hit coordinate. Vertex-format-6 tangent and bitangent directions use the instance linear transform and are Gram-Schmidt orthogonalized at the hit before normal-map application. Dedicated Metal regressions verify authored-normal interpolation and opposite normal-map texels producing different off-axis lighting. A second Metal TLAS intersection from each lit hit supplies SphereLight/DistantLight hard visibility. When `--camera-sensor-output` is requested, the first completed scene frame after live-camera state is available is atomically resampled and published as RGB for the stage extension's one-shot Camera sensor contract. This deliberately avoids a second Replicator Render Product because that unsupported graph produced repeated Render Graph/multiTick/Sensor failures. NVIDIA's Replicator `rgb` CPU array is not claimed as equivalent. Nested instancing and per-instance primvar overrides, animated material inputs, robust skinning/deformation, translated RTX/Hydra material networks, authored tangent/normal-strength semantics, remaining light schemas, soft/area-light shadows, arbitrary Render Products, and native sensor arrays remain incomplete. The ICD must not be installed as a system-wide driver.

## Reproducible validation

Run:

```sh
./scripts/test-vulkan-icd.sh
```

The script builds a Linux ARM64 image, starts the Khronos loader-linked probe in a real Apple container VM, dials its AF_VSOCK listener through Apple's `ContainerClient`, and requires device discovery, storage-buffer and R32_UINT texel-buffer compute, transfer/image readback, sparse-image map/unmap, raster, KHR BLAS/TLAS, OPAQUE_FD round-trip, and `VULKAN_RAY_DISPATCH rays=64x64 center=hit corner=miss backend=Metal fence=signaled`. Test containers are removed and a builder started by the script is stopped on exit.

The real Full/UI milestone is reproducible with `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid`. Use `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --demo-scene --animate-demo --no-build` for the bounded live-timeline path, or add `--camera-sensor-output build/runtime/camera-rgb.ppm` for the one-shot bounded RGB check. The fixed-raster test writes checked PNG artifacts under `build/artifacts`; the live viewer reads atomic UI frame captures under `build/runtime` and appends input records that the real Kit input provider consumes. The next Vulkan milestones are material/shader semantics, additional formats/descriptors, nested instancing, robust skinning/deformation, and native RTX annotator arrays. Capability advertisement must expand only as each call path is executed and validated.
