#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"

swift build --package-path "${repo_root}/host/ContainerAdapter" -c release

echo "build-container-adapter: built imb-container-host with Apple's ContainerAPIClient"
