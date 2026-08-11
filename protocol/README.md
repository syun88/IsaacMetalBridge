# IsaacMetalBridge protocol v1

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Encoding

The protocol is a binary, request/reply stream. All integers are unsigned little-endian. Each message is one packed 32-byte header followed by exactly `payload_length` bytes. Payloads larger than 16 MiB are rejected in v1. Strings are UTF-8 byte sequences without a trailing NUL.

The magic value is `0x31424d49`, which appears as `IMB1` on the wire. Version 1.21 is the only implemented version. It retains Metal sparse-image tile mappings and generic compute descriptor records, inline independent Vulkan sampler state, image formats 8/9/10, tightly packed ordinary 3D images, and bounded camera/light ray submissions from earlier versions. Image format 11 adds sparse RGBA32 float storage used by Isaac's IES texture processor. Positional records preserve SphereLight directly, carry RectLight/DiskLight world axes and half extents, CylinderLight axial/radial geometry, and an explicitly flagged `UsdLuxShapingAPI` tail. Version 1.21 retains `TRACE_RAYS` option bit 4 for the renderer-owned empty-stage guide: its acceleration-structure resource ID is zero, light bits are forbidden, and the live camera remains optional. The same command can be submitted more than once for viewport and Camera Render Product traces; scene-state v19 Mesh geometry, authored-normal, corner-UV, bounded RGBA8 material/light images, roughness, metallic, emission, standard opacity, and opacityThreshold data plus sensor files remain per-run sidebands.

`TRACE_RAYS` option bit 5 carries a complete light-list tail. The unchanged 168-byte base command is followed by an 8-byte count/reserved header and at most sixteen 152-byte records. The first 80 bytes retain the v1.19 kind/schema, sixteen values, and stable USD path hash. RectLight/DiskLight use the value fields for two transformed in-plane axes and their half extents. CylinderLight stores transformed local Z radius in value 7, transformed local X unit axis in values 8-10, half length in value 11, transformed local Y unit axis in values 12-14, and transformed local Y radius in value 15; its third axis is derived by `cross(axisX, axisY)`. `treatAsLine` sets both radial extents to zero.

Bytes 80-119 carry a normalized world emission axis, cone cutoff angle in degrees, cone softness, focus exponent, focus tint, and a 32-bit shaping flag. Bit zero means `UsdLuxShapingAPI` was explicitly applied; all shaping values must be zero otherwise. Metal evaluates the OpenUSD cone smooth region, `abs(dot(axis, emissionDirection))^focus`, and component-wise focus-tint interpolation for every positional-light sample. Bytes 120-151 add RectLight-emission and IES-profile image resource IDs, finite `ies:angleScale` and precomputed IES multiplier values, a light-texture flag word, and reserved zero. Flag bit 0 enables a RectLight sRGB emission image, bit 1 enables a linear angular IES LUT on a shaped positional light, and bit 2 records authored `ies:normalize`. The bounded stage decoder accepts local LM-63 Type C data with `TILT=NONE` or inline `TILT=INCLUDE`, applies the OpenUSD angle-scale mapping, and bakes normalization into the multiplier. Scene-state v19 keeps the 160-byte fixed header, appends the 152-byte complete light list and its bounded RGBA8 light-image table, and admits up to eight positional, four distant, and four dome lights total. Scene-state v3-v18 remains readable by the ICD and is upgraded with zero light-texture fields.

Metal evaluates ordinary CylinderLight records with two axial positions and four radial angles (eight samples total). Each sample uses the elliptical-cylinder gradient for an outward world-space surface normal and its own opacity-aware shadow traversal. A `treatAsLine` record uses two axial samples and no surface-facing rejection. This bounded encoding preserves ordinary nonuniform radial scale; arbitrary shear is not represented exactly.

`DistantLight` value 7 is OpenUSD's full angular diameter in degrees. Receivers accept finite values in `[0, 360)`; zero is a perfectly parallel source and nonzero values drive the bounded four-direction spherical-cap sampling performed by Metal.

```text
offset  size  field
0       4     magic
4       2     version_major
6       2     version_minor
8       2     message_type
10      2     flags
12      4     payload_length
16      8     request_id
24      8     resource_id
```

`request_id` is selected by the requester and echoed by the reply. Zero is invalid for requests. `resource_id` is selected by the host and is scoped to one connection/session. The response flag is set on every reply.

## Negotiation

The first request must be `HELLO` with an inclusive minimum and maximum version. The host selects a compatible version and replies with `HELLO_REPLY`; otherwise it returns `ERROR/UNSUPPORTED_VERSION`. No fallback to an unnegotiated version is permitted.

`QUERY_CAPABILITIES` returns only observed host capability facts and implemented protocol capabilities. `METAL_BUFFER`, `METAL_COMPUTE`, `RESOURCE_IO`, `REAL_FENCE`, `METAL_IMAGE`, `METAL_RASTER`, and `METAL_UI_RASTER` are advertised only when the live Metal backend was created. `METAL_SPIRV_COMPUTE` additionally requires an executable configured SPIRV-Cross translator. `METAL_ACCELERATION_STRUCTURE` and `METAL_RAY_DISPATCH` require a live Metal device with ray-tracing support and the corresponding backend pipelines. These bits do not imply general Vulkan/CUDA/RTX compatibility.

## Messages

| Message | Plane | Request/reply behavior |
|---|---|---|
| `HELLO`, `HELLO_REPLY` | Control | Negotiate protocol version. |
| `QUERY_CAPABILITIES`, `CAPABILITIES_REPLY` | Control | Return bits, max Metal buffer length, and GPU name. |
| `PING`, `PONG` | Control | Echo arbitrary payload. |
| `CREATE_RESOURCE` | Control | Allocate a shared Metal buffer, an ordinary 2D/3D image, or a sparse image; reply carries its session-scoped resource ID. Ordinary 3D uses option bit 1 and a 16-bit depth in option bits 16-31. |
| `QUERY_SPARSE_IMAGE_PROPERTIES` | Control | Return the Metal-backed sparse tile dimensions and mip-tail boundary for a known sparse image. |
| `UPDATE_SPARSE_IMAGE_MAPPING` | Data/control boundary | Map or unmap a validated Vulkan image-tile range in the image's real Metal sparse heap. |
| `CREATE_COMPUTE_PIPELINE` | Control/data | Validate SPIR-V bytes and an entry-point name, translate to MSL, compile a real Metal compute pipeline, and return its session-scoped resource ID. The optional four-byte reply payload is a little-endian creation-flags word: bit 0 (`SOFTWARE_FP64_EXECUTION_REQUIRED`) marks selected-entry-point software binary64 operations, and bit 1 (`SERIALIZED_ATOMIC64_EXECUTION_REQUIRED`) marks a coordination-free 64-bit compare-exchange kernel whose complete logical grid must execute on one Metal thread. An empty reply remains accepted conservatively for compatibility. |
| `CREATE_ACCELERATION_STRUCTURE` | Control/data | Create a session-owned primitive or instance Metal acceleration-structure resource. |
| `BUILD_PRIMITIVE_ACCELERATION_STRUCTURE` | Data | Build triangle geometry into a primitive Metal acceleration structure from bounded buffer resources. Triangle vertex format `1` is `float3` position; format `2` interleaves `float3` position and `float3` authored corner normal; format `3` adds `float2` corner UV; format `4` adds scalar roughness and metallic; format `5` adds float3 emissive color and scalar intensity for host-side bounded material shading. |
| `BUILD_INSTANCE_ACCELERATION_STRUCTURE` | Data | Build a Metal instance acceleration structure from transforms and session-owned primitive structures. |
| `WRITE_RESOURCE` | Data | Copy a bounded byte range into a known buffer. Payload is offset, byte length, reserved zero, then bytes. |
| `READ_RESOURCE` | Data | Read a bounded byte range from a known buffer, or one complete tightly packed image; reply payload is the bytes. |
| `DESTROY_RESOURCE` | Control | Release a real Metal buffer, image, or translated compute pipeline and its session resource ID. |
| `SUBMIT_COMMAND` | Data/control boundary | Accept `NOOP`, `ADD_U32`, fixed `DRAW_TRIANGLE`, `DRAW_INDEXED_UI`, `TRACE_RAYS`, or generic `DISPATCH_COMPUTE`; supported buffer/image/texel-buffer/sampler bindings and push constants target known resources and reply with a real Metal command-buffer fence ID. Sampler kind 7 has resource ID zero, carries packed filter/address/compare options in `format`, min/max LOD float bits in `offset`, and mip-bias/max-anisotropy float bits in `length`. `TRACE_RAYS` may carry validated camera position, forward/up basis, vertical FOV, near/far clip values, and bounded positional-, distant-, and dome-light records. The positional wire record is native for SphereLight; RectLight/DiskLight carry transformed axes and half extents for four-sample, one-sided area lighting, while CylinderLight carries axial/radial geometry for eight outward surface samples or two `treatAsLine` samples. Applied ShapingAPI records add cone, softness, focus, focusTint, and bounded Type C IES response; RectLight records may also reference an sRGB emission image. Option bit 4 instead requests the zero-AS empty-stage guide grid. |
| `WAIT_FENCE` | Control | Wait for a known Metal command buffer and return its completion state. |
| `ERROR` | Control | Error code and UTF-8 explanation; uses the failed request ID. |
| `SHUTDOWN` | Control | Reply, release session state, and close. |

## Control and data planes

Version 1.21 implements a deliberately bounded data plane: each control-stream payload is capped at 16 MiB, resource bounds are checked, and resource/fence ownership is session-scoped. Isaac Kit's larger ring-buffer ranges are transferred as multiple bounded writes. Compute-pipeline and acceleration-structure creation are live; the former returns the optional execution-classification flags described above. `TRACE_RAYS` intersects a known TLAS into RGBA8 and can use the active Kit camera plus bounded positional/Distant/Dome light arrays for viewport or Camera Render Product dimensions; the option-bit-4 form draws the empty-stage grid with resource ID zero and no scene lights. Sparse-image mappings, including RGBA32 float, reach Metal sparse heaps, and `DISPATCH_COMPUTE` carries push constants plus buffer, image, formatted texel-buffer, and independent sampler descriptors. Vulkan sets 0-7 use Metal argument buffers, while sets 8 and above use validated direct Metal resource indices. Push constants are matched by the SPIR-V `PushConstant` variable result ID/name to the exact translated Metal constant parameter, preventing an earlier direct UBO from receiving those bytes. Atomic scene-state v19 independently carries authored visible-Mesh points, triangulated indices, bounds, material flags, transforms, normalized per-corner normals, per-corner float2 UVs, globally deduplicated bounded RGBA8 base/parameter/emission/normal images, a separate bounded light-image table, bounded direct material constants, standard opacity/opacityThreshold, direct emissive color/intensity, and the complete shaped-light-capable list. Native USD scenegraph instance proxies and bounded PointInstancer occurrences are expanded as ordinary per-path Mesh records. Normal and UV values resolve all five USD Mesh/primvar interpolation modes; the guest bakes a standard `UsdTransform2d` scale-rotate-translate chain into those corner UVs. The guest creates separate images and validated 48-byte `MBM1 v1`, 56-byte `MBM1 v2`, or 64-byte `MBM1 v3` material descriptors. v3 distinguishes standard opacity and base-alpha sampling so unrelated texture alpha remains opaque. Vertex format 6 adds normalized tangent/bitangent vectors derived from real triangle positions and transformed UV derivatives. The host binds at most 126 unique material and light images, transforms and orthogonalizes the TBN basis, samples repeat-wrapped tangent-space normals and RectLight emission, applies the bounded Type C IES response, applies diffuse/specular response plus self-emission, source-over composites up to 64 standard-opacity layers in linear color when `opacityThreshold == 0`, and applies the same cutout/fractional transmittance to positional-light/CylinderLight/DistantLight shadow rays. Scene-state v3-v18 remains accepted. General material networks, colored absorption and refraction/volume/thin-walled semantics, translated ray-shader semantics, animated/nested instancing, per-instance primvar overrides, skinning/deformation, authored tangent primvars, different UV sources within one material, UDIM/mip generation, remaining Geometry/Portal light schemas, DomeLight HDR textures, LM-63 Type A/B or external TILT data, unsupported/partially-bound RTX descriptor patterns, arbitrary software-FP64 modules beyond the fourteen exact captured fixtures, 64-bit atomics requiring coordination or operations other than the validated compare-exchange, shared memory, and bulk queues remain future work.

## Validation and errors

Receivers validate magic, exact supported version after negotiation, message type, response direction, payload length, fixed payload sizes, reserved-zero fields, resource existence/type, buffer ranges, and command support. Unknown messages and fields are not silently executed. An invalid stream header that cannot be safely correlated may terminate the connection.

The canonical ABI is [include/imb_protocol.h](include/imb_protocol.h). Changing any packed layout or semantic requires a protocol version change and cross-language tests.
