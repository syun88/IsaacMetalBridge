# Contributing to IsaacMetalBridge

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Principles

- Preserve the real Isaac Sim Linux environment; do not add substitute simulators or fake Isaac APIs.
- Do not claim GPU, rendering, CUDA, Vulkan, or Isaac Sim functionality without reproducible evidence.
- Do not commit Isaac Sim packages, OCI archives, assets, caches, credentials, logs containing secrets, or other large/proprietary artifacts.
- Keep `third_party/apple-container` pinned and clean. Put required upstream changes in `patches/apple-container/` with an explanation and exact source references.
- Keep control-plane and data-plane changes separate and version the wire protocol before changing its layout.

## Development workflow

1. Run `./scripts/doctor.sh`.
2. Build with `./scripts/build-all.sh`.
3. Run `./scripts/test-all.sh`.
4. For transport or GPU-data-path changes, also run `./scripts/test-all.sh --container`.
5. Document what was actually executed, including failures and unavailable hardware.

The repository owner has not selected a project license. Discuss licensing before submitting code intended for redistribution.
