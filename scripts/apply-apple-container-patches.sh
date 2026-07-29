#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
submodule="${repo_root}/third_party/apple-container"
patch_dir="${repo_root}/patches/apple-container"
patches=("${patch_dir}"/*.patch)

if [[ ! -d "${submodule}" ]]; then
    echo "apply-apple-container-patches: initialize the submodule first" >&2
    exit 1
fi
if [[ -n "$(git -C "${submodule}" status --short)" ]]; then
    echo "apply-apple-container-patches: submodule must be clean before applying patches" >&2
    exit 1
fi
if [[ ! -e "${patches[0]}" ]]; then
    echo "apply-apple-container-patches: no patch files present"
    exit 0
fi

for patch_file in "${patches[@]}"; do
    git -C "${submodule}" apply --check "${patch_file}"
done
for patch_file in "${patches[@]}"; do
    git -C "${submodule}" apply "${patch_file}"
    echo "applied $(basename "${patch_file}")"
done
