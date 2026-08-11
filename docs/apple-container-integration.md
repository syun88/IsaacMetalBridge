# Apple container integration

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Audited revisions

- `apple/container`: `14233cee65486c1ada2b82403c17d1236a9176c2`
- pinned `apple/containerization` dependency: version `0.37.0`, revision `1d5641ff962456df021283505eaa6a701d828192`

The first revision is the Git submodule. The second is recorded in the submodule's `Package.resolved`. Findings below come from these exact sources.

## Lifecycle and source map

| Area | Source path | Type/function | Verified role |
|---|---|---|---|
| API lifecycle | `Sources/Services/ContainerAPIService/Server/Containers/ContainersService.swift` | `ContainersService.create`, `bootstrap`, `dial`, `delete` | Persists container state, starts the selected runtime plugin, forwards lifecycle calls, returns the vsock file descriptor. |
| Runtime client | `Sources/Services/Runtime/RuntimeClient/RuntimeClient.swift` | `RuntimeClient.create`, `bootstrap`, `dial` | Opens the per-container XPC endpoint and sends typed routes. |
| Runtime routes | `Sources/Services/Runtime/RuntimeClient/RuntimeRoutes.swift` | `RuntimeRoutes` | Defines lifecycle, process, file, and `dial` routes. |
| Linux helper | `Sources/Plugins/RuntimeLinux/RuntimeLinuxHelper+Start.swift` | `RuntimeLinuxHelper.Start.run` | Creates endpoint and anonymous XPC servers and binds route handlers. |
| VM bootstrap | `Sources/Services/RuntimeLinux/Server/RuntimeService.swift` | `RuntimeService.bootstrap` | Builds bundle/network state, creates `VZVirtualMachineManager`, constructs `LinuxContainer`, and calls `container.create()`. |
| Workload start | same file | `RuntimeService.startProcess`, `startInitProcess` | Starts the real OCI process after the VM is booted. |
| Host-to-guest vsock | same file | `RuntimeService.dial` | Calls `LinuxContainer.dialVsock` and returns a file handle over XPC. |
| Image lifecycle | `Sources/Services/ContainerImagesService/Server/ImagesService.swift` | `pull`, `unpack`, `getImageSnapshot` | Resolves OCI content and snapshots. |
| Snapshot storage | `Sources/Services/ContainerImagesService/Server/SnapshotStore.swift` | `unpack`, `get`, `delete` | Materializes and locates root filesystem snapshots. |
| XPC implementation | `Sources/ContainerXPC/XPCClient.swift`, `XPCServer.swift`, `XPCMessage.swift` | `XPCClient.send`, `XPCServer.listen`, `XPCMessage` | Host process IPC, replies, data, endpoints, and file-descriptor passing. |
| Plugin discovery | `Sources/ContainerPlugin/PluginLoader.swift` | `findPlugins`, `registerWithLaunchd` | Finds configured plugins and registers launchd services. |
| Plugin schema | `Sources/ContainerPlugin/PluginConfig.swift` | `PluginConfig.DaemonPluginType` | Supports runtime, network, core, and auxiliary XPC services. |

## Virtualization.framework configuration

The following paths are in the pinned `apple/containerization` dependency:

| Source path | Type/function | Verified behavior |
|---|---|---|
| `Sources/Containerization/VZVirtualMachineManager.swift` | `VZVirtualMachineManager.create` | Clamps CPU/RAM and creates `VZVirtualMachineInstance`. |
| `Sources/Containerization/VZVirtualMachineInstance.swift` | `Configuration.toVZ` | Creates `VZVirtualMachineConfiguration`, attaches devices, calls extensions, validates configuration. |
| same file | `VZInstanceExtension.configureVZ` | Public pre-creation extension hook for mutating the VZ configuration. |
| same file | `VZVirtualMachineInstance.didCreate/willStop` extension calls | Host lifecycle hooks before VM start and stop. |
| `Sources/Containerization/VZVirtualMachine+Helpers.swift` | `connect`, `listen`, `waitForAgent` | Operates `VZVirtioSocketDevice`; the built-in agent is reached over vsock. |
| `Sources/Containerization/Mount.swift` | `Mount.configure` | Adds VirtIO block storage; VirtioFS is assembled by `Configuration.toVZ`. |
| `Sources/Containerization/LinuxContainer.swift` | `LinuxContainer.create` | Builds the common `VMConfiguration` passed to the VMM. |

`Configuration.toVZ` currently adds:

- `VZVirtioEntropyDeviceConfiguration`
- `VZVirtioSocketDeviceConfiguration`
- VirtIO console/serial output
- `VZVirtioNetworkDeviceConfiguration` instances
- `VZVirtioBlockDeviceConfiguration` for block mounts
- `VZVirtioFileSystemDeviceConfiguration` for shared directories/Rosetta
- `VZGenericPlatformConfiguration`

No `graphicsDevices` assignment or GPU device configuration exists in the audited source.

## Communication mechanisms

### XPC

XPC is host-only control IPC among the CLI/API server, image service, network service, and per-container runtime helper. `XPCMessage` can pass file descriptors and anonymous endpoints.

### Vsock

Vsock is verified. Every VM gets `VZVirtioSocketDeviceConfiguration`. The containerization helper connects to the built-in guest agent over vsock, and Apple container exposes `ContainerClient.dial(id:port:)` -> `ContainersService.dial` -> `RuntimeClient.dial` -> `RuntimeService.dial` -> `LinuxContainer.dialVsock`.

For finite `--/app/quitAfter` runs, Kit may close that full-duplex stream after reading a final request but before the host writes its response. `imb-container-host --wait-exit` recognizes only the resulting EOF/EPIPE/connection-reset cases and accepts them only after `ClientProcess.wait()` independently returns init status zero; malformed protocol data and nonzero exits still fail the launcher.

The public Apple-container route is **host-dials-guest**. It does not expose a guest-initiated host listener. The first production IMB transport should therefore run a guest listener and let `imb-host` obtain the connected file descriptor through this route. Any reverse direction requires a separately reviewed change.

### VirtIO and shared memory

VirtIO block, network, socket, console, and filesystem devices are verified. A dedicated GPU VirtIO device and cross-VM shared-memory transport are not present. VirtioFS copies/shares files; it is not evidence of coherent GPU command memory.

## GPU bridge insertion locations

1. **Preferred VM hook:** implement a `VZInstanceExtension` whose `configureVZ` adds only a supported, documented VZ device and whose lifecycle methods own host resources. This hook is before `VZVirtualMachineConfiguration.validate()`.
2. **Required Apple-container wiring:** the audited `LinuxContainer.create()` does not forward an extension from `RuntimeService.bootstrap`. A narrow wrapper around `VZVirtualMachineManager` at the construction site in `RuntimeService.bootstrap`, or an upstream `LinuxContainer.Configuration.extensions` property, must append the extension to the `VMConfiguration`.
3. **Host service:** a core/auxiliary Apple-container plugin can host command decoding and Metal resources via XPC. This is useful process isolation but does not itself attach a VM device.
4. **Implemented no-patch transport:** `imb-container-host` imports `ContainerAPIClient`, calls `ContainerClient.dial(id:port:)`, and serves IMB over the returned full-duplex `FileHandle`. The Linux probe binds `AF_VSOCK`/`VMADDR_CID_ANY`. Protocol 1.18, real Metal sparse-image mappings, generic buffer/image/texel-buffer/sampler compute, translated SPIR-V pipeline creation, fixed raster, Kit UI raster, the camera-aware empty-stage guide, bounded visible-USD-Mesh-matched Metal acceleration structures, active-viewport-camera ray dispatch and RGB sensor publication, live positional-light/DistantLight/DomeLight state, fences, and frame output passed through this path without modifying Apple source. SphereLight is direct; RectLight/DiskLight carry world axes and half extents for bounded four-sample area lighting.
5. **Future bulk data:** only after measurement, investigate a supported VZ memory/file-sharing mechanism. Do not label VirtioFS as shared GPU memory.

## Patch policy

The submodule remains read-only. If device wiring becomes necessary, identify exact source functions above, generate a patch against the pinned commit, store it under `patches/apple-container/`, and apply/reverse it with the repository scripts. No Apple-source patch is required for the verified vsock/Metal path.
