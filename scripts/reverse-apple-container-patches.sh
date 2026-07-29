#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
submodule="${repo_root}/third_party/apple-container"
patch_dir="${repo_root}/patches/apple-container"
patches=("${patch_dir}"/*.patch)

if [[ ! -d "${submodule}" ]]; then
    echo "reverse-apple-container-patches: initialize the submodule first" >&2
    exit 1
fi
if [[ ! -e "${patches[0]}" ]]; then
    echo "reverse-apple-container-patches: no patch files present"
    exit 0
fi

for ((index=${#patches[@]} - 1; index >= 0; index--)); do
    git -C "${submodule}" apply --reverse --check "${patches[index]}"
done
for ((index=${#patches[@]} - 1; index >= 0; index--)); do
    git -C "${submodule}" apply --reverse "${patches[index]}"
    echo "reversed $(basename "${patches[index]}")"
done

if [[ -n "$(git -C "${submodule}" status --short)" ]]; then
    echo "reverse-apple-container-patches: submodule still has tracked modifications" >&2
    exit 1
fi
