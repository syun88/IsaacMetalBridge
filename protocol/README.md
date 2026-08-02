# IsaacMetalBridge protocol v1

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Encoding

The protocol is a binary, request/reply stream. All integers are unsigned little-endian. Each message is one packed 32-byte header followed by exactly `payload_length` bytes. Payloads larger than 16 MiB are rejected in v1. Strings are UTF-8 byte sequences without a trailing NUL.

The magic value is `0x31424d49`, which appears as `IMB1` on the wire. Version 1.13 is the only implemented version. It retains Metal sparse-image tile mappings and generic compute descriptor records, including Vulkan-format texel buffers, and extends bounded ray submissions with optional live Kit camera plus first USD SphereLight, DistantLight, and DomeLight records. The same command can be submitted more than once for viewport and Camera Render Product traces; scene-state v11 Mesh geometry, authored-normal, corner-UV, bounded RGBA8 base/parameter/normal-image, roughness, metallic, and emission data plus sensor files remain per-run sidebands.

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
| `CREATE_RESOURCE` | Control | Allocate a shared Metal buffer, a linear image, or a sparse image; reply carries its session-scoped resource ID. |
| `QUERY_SPARSE_IMAGE_PROPERTIES` | Control | Return the Metal-backed sparse tile dimensions and mip-tail boundary for a known sparse image. |
| `UPDATE_SPARSE_IMAGE_MAPPING` | Data/control boundary | Map or unmap a validated Vulkan image-tile range in the image's real Metal sparse heap. |
| `CREATE_COMPUTE_PIPELINE` | Control/data | Validate SPIR-V bytes and an entry-point name, translate to MSL, compile a real Metal compute pipeline, and return its session-scoped resource ID. |
| `CREATE_ACCELERATION_STRUCTURE` | Control/data | Create a session-owned primitive or instance Metal acceleration-structure resource. |
| `BUILD_PRIMITIVE_ACCELERATION_STRUCTURE` | Data | Build triangle geometry into a primitive Metal acceleration structure from bounded buffer resources. Triangle vertex format `1` is `float3` position; format `2` interleaves `float3` position and `float3` authored corner normal; format `3` adds `float2` corner UV; format `4` adds scalar roughness and metallic; format `5` adds float3 emissive color and scalar intensity for host-side bounded material shading. |
| `BUILD_INSTANCE_ACCELERATION_STRUCTURE` | Data | Build a Metal instance acceleration structure from transforms and session-owned primitive structures. |
| `WRITE_RESOURCE` | Data | Copy a bounded byte range into a known buffer. Payload is offset, byte length, reserved zero, then bytes. |
| `READ_RESOURCE` | Data | Read a bounded byte range from a known buffer, or one complete tightly packed image; reply payload is the bytes. |
| `DESTROY_RESOURCE` | Control | Release a real Metal buffer, image, or translated compute pipeline and its session resource ID. |
| `SUBMIT_COMMAND` | Data/control boundary | Accept `NOOP`, `ADD_U32`, fixed `DRAW_TRIANGLE`, `DRAW_INDEXED_UI`, `TRACE_RAYS`, or generic `DISPATCH_COMPUTE`; supported buffer/image/texel-buffer bindings and push constants target known resources and reply with a real Metal command-buffer fence ID. `TRACE_RAYS` may carry validated camera position, forward/up basis, vertical FOV, near/far clip values, and first USD SphereLight, DistantLight, and DomeLight records. |
| `WAIT_FENCE` | Control | Wait for a known Metal command buffer and return its completion state. |
| `ERROR` | Control | Error code and UTF-8 explanation; uses the failed request ID. |
| `SHUTDOWN` | Control | Reply, release session state, and close. |

## Control and data planes

Version 1.13 implements a deliberately bounded data plane: each control-stream payload is capped at 16 MiB, resource bounds are checked, and resource/fence ownership is session-scoped. Isaac Kit's larger ring-buffer ranges are transferred as multiple bounded writes. Compute-pipeline and acceleration-structure creation are live; `TRACE_RAYS` intersects a known TLAS into RGBA8 and can use the active Kit camera plus the first SphereLight/DistantLight/DomeLight for viewport or Camera Render Product dimensions; sparse-image mappings reach Metal sparse heaps; and `DISPATCH_COMPUTE` carries push constants plus buffer, image, and formatted texel-buffer descriptors. Atomic scene-state v11 independently carries authored visible-Mesh points, triangulated indices, bounds, material flags, transforms, normalized per-corner normals, per-corner float2 UVs, optional RGBA8 base-color/roughness/metallic/emission/normal pixels up to 4096x4096, bounded direct material constants, and direct emissive color/intensity. Native USD scenegraph instance proxies are expanded as separate Mesh records with their composed paths and world transforms. Default-time `UsdGeomPointInstancer` prototypes are likewise expanded after OpenUSD evaluates prototype indices, SRT, IDs, and visibility masks, so no wire change is required. Normal and UV values resolve all five USD Mesh/primvar interpolation modes; the guest bakes a standard `UsdTransform2d` scale-rotate-translate chain into those corner UVs. Images share one complete UV source in v11; scalar output-channel selectors and RGB emission/normal data are retained. The guest creates separate images and a validated 48-byte `MBM1 v1` or 56-byte `MBM1 v2` material descriptor. Vertex format 6 adds normalized tangent/bitangent vectors derived from real triangle positions and transformed UV derivatives. The host binds at most 16 unique images, transforms and orthogonalizes the TBN basis, samples repeat-wrapped tangent-space normals, applies bounded diffuse/specular response plus self-emission, and traces SphereLight/DistantLight hard-shadow rays. Scene-state v3-v10 remains accepted. General material networks/translated ray-shader semantics, animated/nested instancing, per-instance primvar overrides, skinning/deformation, authored tangent primvars, normal strength, different UV sources within one material, UDIM/mip generation, soft/area-light shadows, remaining light schemas, unsupported/partially-bound RTX descriptor patterns, FP64 lowering, shared memory, and bulk queues remain future work.

## Validation and errors

Receivers validate magic, exact supported version after negotiation, message type, response direction, payload length, fixed payload sizes, reserved-zero fields, resource existence/type, buffer ranges, and command support. Unknown messages and fields are not silently executed. An invalid stream header that cannot be safely correlated may terminate the connection.

The canonical ABI is [include/imb_protocol.h](include/imb_protocol.h). Changing any packed layout or semantic requires a protocol version change and cross-language tests.
