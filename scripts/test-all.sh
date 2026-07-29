#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
run_container=0

if [[ "${1:-}" == "--container" ]]; then
    run_container=1
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--container]" >&2
    exit 2
fi

"${script_dir}/build-all.sh"
swift test --package-path "${repo_root}/host"
ctest --test-dir "${repo_root}/build/guest" --output-on-failure -C Release
"${script_dir}/run-guest-probe.sh"

if [[ "${run_container}" -eq 1 ]]; then
    "${script_dir}/build-container-adapter.sh"
    "${script_dir}/test-container-vsock.sh"
    IMB_SKIP_GUEST_BUILD=1 "${script_dir}/test-vulkan-icd.sh"
fi

if [[ -n "$(git -C "${repo_root}/third_party/apple-container" status --short)" ]]; then
    echo "test-all: Apple container submodule is dirty" >&2
    exit 1
fi

if [[ "${run_container}" -eq 1 ]]; then
    echo "test-all: unit, ABI, local, Apple-container/vsock/Metal, and Vulkan ICD tests completed"
else
    echo "test-all: unit, ABI, lifecycle, and local cross-language tests completed"
fi
