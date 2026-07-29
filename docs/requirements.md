# Requirements

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Development host

- Apple Silicon Mac (`arm64`)
- macOS 26 recommended by the pinned Apple `container` source; macOS 15 is its build minimum
- Swift 6 or newer and the Metal SDK
- CMake 3.25 or newer
- C++20 compiler
- Git with submodule support
- Network access for the first `ContainerAPIClient` Swift dependency resolution and Alpine guest-image build

## Runtime validation inputs

- Apple `container` installed and its system service running
- A Linux ARM64 guest/image
- Real Isaac Sim 6.0.1 Linux aarch64 content stored outside this repository
- Sufficient external disk space (NVIDIA advises at least 50 GB free for standalone quick installation)
- License acceptance and registry credentials managed by the user, never by repository scripts

## Current host observation (2026-07-19)

Verified on the development host: Apple Silicon `arm64`, macOS 27.0, Swift 6.4, CMake 4.1.0, Apple Clang 21.0.0, and Apple `container` 1.1.0. A real `linux/arm64` guest returned `aarch64`. The official Isaac Sim 6.0.1 ARM64 image is loaded in Apple container's image store and selected by the ignored local `.env`. The Apple client adapter built against the pinned submodule, and a real guest-vsock session completed Metal buffer compute plus fixed RGBA8 raster, fences, and readback.

These observations are a point-in-time result, not a general support claim.
