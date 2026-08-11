# Architecture

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

Rendering blocks distinguish the verified fixed-triangle and Kit UI slices from future RTX/general rendering.

## 1. Full system architecture

```mermaid
flowchart TB
  subgraph macOS["Apple Silicon macOS"]
    AC["Apple container services"]
    HB["imb-host"]
    MB["Metal backend"]
    GPU["Apple GPU"]
    VW["Native macOS viewer"]
  end
  subgraph VM["Linux ARM64 VM"]
    IS["Real Isaac Sim"]
    KIT["Omniverse Kit / Carbonite / USD / PhysX"]
    GC["Guest Vulkan/CUDA compatibility layers"]
    GP["IMB guest endpoint"]
  end
  IS --> KIT --> GC --> GP
  AC --> VM
  GP <-->|"IMB 1.17 over host-dialed vsock (verified)"| HB
  HB --> MB --> GPU
  MB -->|"atomic real Kit UI frames"| VW
```

## 2. Isaac Sim execution path

```mermaid
sequenceDiagram
  participant CLI as container CLI/API
  participant RT as RuntimeService
  participant VZ as Linux VM
  participant Kit as Isaac Sim/Kit
  participant Bridge as IMB compatibility layer
  CLI->>RT: create + bootstrap
  RT->>VZ: configure and start VM
  RT->>VZ: start OCI process
  VZ->>Kit: execute real ARM64 launcher
  Kit->>Bridge: load/call GPU-facing API
  Bridge-->>Kit: implemented result or explicit unsupported error
```

## 3. GPU command path

```mermaid
flowchart LR
  API["IMB probe / Vulkan / Kit UI / CUDA startup calls"] --> VAL["Guest validation"]
  VAL --> ENC["IMB command encoder"]
  ENC --> Q["Transport queue"]
  Q --> DEC["Host decoder"]
  DEC --> CAP["Capability manager"]
  DEC --> RM["Resource manager"]
  DEC --> METAL["Metal command encoder (generic compute + sparse image + bounded raster/ray verified)"]
  METAL --> CQ["MTLCommandQueue"]
```

## 4. Host/guest communication

```mermaid
flowchart LR
  G["Linux AF_VSOCK listener / probe"] --> CP["Control plane"]
  CP -->|"HELLO, capabilities, resources, errors"| FR["IMB framing"]
  DP["Data plane"] -->|"buffers + images + sparse tiles + compute descriptors + Kit UI draws"| FR
  FR <-->|"local pipes and Apple host-dialed vsock verified"| H["Swift host session"]
```

Apple's audited API supports the host dialing a guest vsock port. `imb-container-host` now uses that exact API and has completed a real ARM64 VM session. Shared memory and a custom VirtIO GPU remain unverified mechanisms.

## 5. Vulkan bridge prototype

```mermaid
flowchart LR
  VKL["Guest Vulkan loader / real Kit"] --> ICD["IMB Vulkan startup ICD + supported Metal execution paths"]
  ICD --> CAPS["Truthful Vulkan capability filter"]
  ICD --> SPV["SPIR-V translation + buffer/image/texel dispatch"]
  ICD --> SPR["Sparse image tile requirements + mappings"]
  ICD --> UI["Exact tested Kit UI pipeline recognition"]
  ICD --> P["IMB protocol"]
  P --> MSL["MSL/shader library"]
  P --> MR["Metal resources and commands"]
```

The ICD must advertise only implemented limits/features and return standard errors for unsupported behavior.

## 6. CUDA startup bridge and future execution path

```mermaid
flowchart LR
  APP["Kit / PhysX / Warp"] --> CUD["Guest CUDA Driver ABI"]
  CUD --> CTX["Context/module/memory model"]
  CTX --> PTX["PTX strategy: unresolved"]
  CTX --> STR["Streams/events mapping"]
  PTX --> IMB["IMB data/control plane"]
  STR --> IMB
  IMB --> MET["Metal compute"]
```

The tested context/memory/stream/event/module and private export-table startup calls are implemented locally in the guest shim. CUDA kernel execution, PTX conversion, and Metal stream semantics remain unresolved; GPU PhysX may remain blocked.

## 7. Resource lifecycle

```mermaid
stateDiagram-v2
  [*] --> Negotiated: HELLO
  Negotiated --> Allocated: CREATE_RESOURCE
  Allocated --> Submitted: SUBMIT_COMMAND
  Submitted --> Signaled: WAIT_FENCE
  Signaled --> Allocated: reuse
  Allocated --> Destroyed: DESTROY_RESOURCE
  Destroyed --> [*]: SHUTDOWN/disconnect
```

Resource and fence IDs are host-issued. Disconnect destroys session-owned state. Protocol 1.17 buffer resources are real shared `MTLBuffer` objects; linear and sparse images are real `MTLTexture` objects, with sparse images owning Metal heaps and tile mappings including RGBA8 sRGB, BC3 sRGB, and BC5 UNORM; translated compute pipelines are real `MTLComputePipelineState` objects; primitive/instance acceleration resources own real Metal acceleration structures; and compute, raster, UI, sparse-map, and ray fences correspond to committed Metal command buffers. Generic compute bindings carry buffers, images, formatted texel-buffer views, independent sampler state, and push constants. Ray submissions can carry a validated live Kit camera and bounded positional-, distant-, and dome-light records. The positional wire record preserves SphereLight or carries a guest-generated center-point/equal-area-radius hard approximation of RectLight/DiskLight. The option-bit-4 zero-AS ray form renders the perspective empty-stage guide and axes from that camera into Kit's landscape viewport images without replacing the USD stage or its UI atlas. A separate atomic scene-state v15 sideband adds each visible USD Mesh's real points, triangulated indices, normalized authored corner normals, optional corner UVs, direct material constants, standard opacity/opacityThreshold, bounds, flags, transform, and five indices into a globally deduplicated RGBA8 base/roughness/metallic/emission/normal image table so the guest can build deterministic per-Mesh Metal BLASes; v3-v14 parsing and v3/v4 bounds matching remain backward compatible. `materialBind` GeomSubsets become disjoint face records. Traversal enables native scenegraph instance proxies. The extension converts `omni.timeline` seconds to stage time codes and asks OpenUSD to evaluate camera, transforms, Mesh attributes, lights, proxy transforms, and PointInstancer prototype indices, positions, quaternion orientations, scales, IDs, masks, prototype-root transforms, and instancer transforms at that current time. Each visible prototype Mesh is emitted with a synthetic stable path and composed matrix. Static changes advance the Mesh/light/material sequence and rebuild the fallback TLAS; camera-only changes advance a separate camera sequence and reuse the TLAS from the 160-byte header. A content hash excluding path and world transform lets transform-only occurrences reuse the same Metal BLAS/material resources after static updates. The guest resolves standard `UsdTransform2d` chains and bakes scale-rotate-translate into corner UVs before transport. Vertex format 6 appends tangent and bitangent vectors derived from real triangle positions and transformed UV derivatives. The host transforms three normals, barycentrically interpolates normals/UVs, transforms and orthogonalizes the TBN basis, and retains two float4 direct-material records per TLAS instance. Parameter images use a 48-byte `MBM1 v1` descriptor with four host image IDs; `MBM1 v2` is 56 bytes and adds the normal image plus a bounded legacy alpha-cutout flag; `MBM1 v3` is 64 bytes and adds explicit standard-opacity/base-alpha flags plus opacity and opacityThreshold. The host validates the descriptor before binding up to 126 unique textures. Tagged Warehouse Barcode/WallBoard intersections sample base alpha first and advance the ray through values below `0.3333`. Standard v3 opacity uses the authored threshold and optional base alpha: positive thresholds perform cutout re-intersection, while zero-threshold fractional hits are shaded and source-over composed front-to-back in linear color. Positional-light/DistantLight shadow rays make the same decision and accumulate transmittance. Both primary and shadow paths have a 64-intersection cap. Base/emission texture samples use sRGB-to-linear transfer before lighting and completed linear output is encoded to sRGB. The real Simple Grid texture is loaded directly by the host from the launcher's asset cache. The host can atomically capture completed Kit UI targets for the native viewer. Unsupported broader descriptor/material graphs, colored absorption and refraction/volume/thin-walled semantics, authored tangents, different UV sources within one material, orientation-dependent area emission, nested instancing and per-instance primvar overrides, animated material inputs, robust skinning/deformation, and general RTX/Hydra shader semantics remain incomplete.

## 8. Rendering flow

```mermaid
flowchart LR
  KIT["Real Isaac Kit"] --> UI["Recognized UI draw stream"]
  KIT --> KHR["KHR scene BLAS and trace records"]
  KIT --> CAM["Active camera + USD Sphere/Distant/Dome light state"]
  KIT --> USD["Visible USD Mesh points + indices + transforms"]
  USD --> MATCH["Build deterministic per-Mesh Metal BLAS"]
  KHR --> AS["Real Metal primitive and instance AS"]
  MATCH --> AS
  CAM --> RAY["Metal intersection dispatch"]
  AS --> RAY
  RAY --> VP["Kit 1280x720 viewport ring"]
  VP --> SENSOR["Active-camera RGB resample + JSON"]
  VP --> UI
  UI --> MET["Metal indexed UI raster"]
  MET --> CAP["Atomic PPM frame capture"]
  CAP --> VIEW["Native macOS viewer (verified)"]
  VIEW --> EVT["Bounded input-record file"]
  EVT --> INJ["Kit Carbonite input provider"]
  INJ --> KIT
```

## Component boundaries

Guest: the verified probe/listener, Vulkan ICD, CUDA startup shim, timer compatibility shim, and Kit input/stage extensions today; NVML remains future work. Transport: bounded framing and verified vsock for GPU work plus per-run bind-mounted stage/input/scene-state/sensor files, with bulk-transfer mechanisms still future work. Host: decoder, capability manager, real buffer/linear-and-sparse-image/pipeline/acceleration-structure resource manager, real Simple Grid and scene-state material textures/constants, frame capture, and interactive native viewer. Metal: shared buffers, linear and sparse textures, translated compute dispatch, deterministic authored-Mesh, native-instance-proxy, and current-time PointInstancer-expanded primitive acceleration structures plus the refreshable fallback instance structure, per-triangle authored-normal and UV barycentric interpolation with geometric fallback, derived tangent-space basis, file base/roughness/metallic/emission/normal sampling, bounded diffuse-specular shading plus direct or mapped emission, primary and hard-shadow TLAS rays using the live Kit camera and bounded positional/Distant/Dome light arrays, one fixed raster pipeline, and the tested Kit UI pipeline. SphereLight is direct; RectLight/DiskLight are explicit hard center-point approximations. The completed camera image can be resampled into the one-shot RGB contract without adding a second RTX Render Graph. Broader material networks, authored tangents/normal strength, nested instancing and per-instance primvar overrides, animated material inputs, robust skinning/deformation, broader raster/descriptor semantics, true soft/area-light shadows and remaining light schemas, and native RTX annotator/sensor arrays remain future work.
