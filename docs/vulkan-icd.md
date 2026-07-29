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
- a narrow real Metal ray dispatch into RGBA8 and synchronization of the real-Isaac result into Kit's triple-buffered viewport images
- Vulkan sparse-image requirements and tile map/unmap backed by Metal sparse heaps
- real generic compute execution for supported translated pipelines with buffer, image, and formatted texel-buffer bindings
- binary/timeline semaphores and query-pool lifecycle/results
- synchronous validation/no-op handling for generic graphics submissions outside the fixed triangle and recognized Kit UI paths

The ICD accepts a host connection only when `IMB_VSOCK_PORT` is explicitly set. It performs IMB 1.10 negotiation and requires the live host to advertise Metal buffer, compute, image, fixed raster, UI raster, resource-I/O, and real-fence capabilities before exposing the physical device. Live SPIR-V compute-pipeline creation requires `METAL_SPIRV_COMPUTE`; sparse images require `METAL_SPARSE_IMAGE`; acceleration-structure and ray commands require `METAL_ACCELERATION_STRUCTURE` and `METAL_RAY_DISPATCH`.

## Explicit limitations

This is not a conformant Vulkan implementation and does not report a conformance version. Compute shader modules are sent through the protocol to a pinned SPIRV-Cross build and compiled as real Metal pipelines when supported; the current live Isaac trace creates 162 of 204 requested pipelines and rejects 42 FP64-buffer modules explicitly. Supported generic dispatches execute on Metal. Legal null/partially-bound descriptors and unsupported formats are skipped without failing the entire RTX queue; a measured finite Full run executed 595 Metal dispatches and skipped 2,708 such submissions. General graphics pipelines remain compatibility objects/no-ops outside the verified fixed and Kit UI paths.

The verified ray path intersects real scene BLAS geometry, but Kit's NVIDIA-only instance producer leaves the observed TLAS descriptor null. The current compatibility fallback places built BLASes into one identity-transform TLAS and uses a fixed bridge camera plus diagnostic hit/miss shading. This produces a real non-black Metal intersection viewport for the included validation stage, but it does not implement general scene transforms, live Kit camera state, translated RTX/Hydra shaders, materials, lighting, or sensors. The ICD must not be installed as a system-wide driver.

## Reproducible validation

Run:

```sh
./scripts/test-vulkan-icd.sh
```

The script builds a Linux ARM64 image, starts the Khronos loader-linked probe in a real Apple container VM, dials its AF_VSOCK listener through Apple's `ContainerClient`, and requires device discovery, storage-buffer and R32_UINT texel-buffer compute, transfer/image readback, sparse-image map/unmap, raster, KHR BLAS/TLAS, OPAQUE_FD round-trip, and `VULKAN_RAY_DISPATCH rays=64x64 center=hit corner=miss backend=Metal fence=signaled`. Test containers are removed and a builder started by the script is stopped on exit.

The real Full/UI milestone is reproducible with `ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh --experience full --window --simple-grid`. The fixed-raster test writes checked PNG artifacts under `build/artifacts`; the live viewer reads atomic UI frame captures under `build/runtime` and appends input records that the real Kit input provider consumes. The next Vulkan milestones are scene/camera transform propagation, broader formats/descriptors, and RTX shader semantics. Capability advertisement must expand only as each call path is executed and validated.
