# Security Policy

IsaacMetalBridge is an unofficial experimental compatibility layer.
It is not affiliated with, endorsed by, sponsored by, or supported by Apple Inc. or NVIDIA Corporation.

## Prototype warning

The current host/guest protocol is an unauthenticated local research protocol. It is not encrypted, sandboxed, hardened, or suitable for an untrusted network. Message validation reduces accidental corruption; it is not a security boundary.

Do not expose `imb-host` to a network, run untrusted command streams, or use sensitive production data. The no-op command is deliberately the only accepted command.

## Reporting

Report vulnerabilities privately to the repository owner using the security-reporting mechanism configured on the repository host. Do not include credentials, proprietary Isaac Sim files, or exploit payloads in a public issue.

## Secrets and artifacts

Never commit registry tokens, NVIDIA credentials, `.env` files, signing keys, Isaac Sim packages/images, model assets, caches, or diagnostic bundles containing user data.
