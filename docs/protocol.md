# Protocol architecture

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

The canonical v1 wire definition is [../protocol/README.md](../protocol/README.md) and the portable ABI header is [../protocol/include/imb_protocol.h](../protocol/include/imb_protocol.h).

The control plane contains version negotiation, discovery, error reporting, Ping/Pong, shutdown, resources, and fences. Protocol 1.10 includes real Metal buffer allocation, byte upload/readback, linear and sparse images, sparse-tile map/unmap, fixed raster/UI commands, SPIR-V-to-MSL compute-pipeline creation, generic compute dispatch with buffer/image/texel-buffer descriptors and push constants, primitive/instance Metal acceleration structures, bounded ray dispatch, and real Metal command-buffer completion. Translated RTX/Hydra shader semantics, unsupported FP64 modules, bulk queues, and zero-copy sharing remain future work.

The transport remains abstract. Local tests use child-process pipes. Production-path validation uses a Linux AF_VSOCK listener and a macOS adapter that calls Apple container's existing `ContainerClient.dial(id:port:)`; the complete session has passed in a real ARM64 VM. No dedicated VirtIO GPU or coherent shared memory is claimed.
