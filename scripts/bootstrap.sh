#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
expected_apple_commit="14233cee65486c1ada2b82403c17d1236a9176c2"

for required_command in git swift cmake c++; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "bootstrap: missing required command: ${required_command}" >&2
        exit 1
    fi
done

git -C "${repo_root}" submodule update --init --recursive third_party/apple-container
actual_apple_commit="$(git -C "${repo_root}/third_party/apple-container" rev-parse HEAD)"
if [[ "${actual_apple_commit}" != "${expected_apple_commit}" ]]; then
    echo "bootstrap: Apple container commit mismatch" >&2
    echo "  expected: ${expected_apple_commit}" >&2
    echo "  actual:   ${actual_apple_commit}" >&2
    exit 1
fi

if [[ -n "$(git -C "${repo_root}/third_party/apple-container" status --short)" ]]; then
    echo "bootstrap: third_party/apple-container has tracked modifications" >&2
    exit 1
fi

echo "bootstrap: toolchain and pinned Apple container submodule are ready"
echo "bootstrap: no Isaac Sim packages or images were downloaded"
