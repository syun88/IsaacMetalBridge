#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
spirv_cross_commit="6c09849fe88c48eaed08413aa022aaa136a3a057"
source_dir="${IMB_SPIRV_CROSS_SOURCE:-${repo_root}/build/dependencies/spirv-cross-src}"
build_dir="${IMB_SPIRV_CROSS_BUILD:-${repo_root}/build/dependencies/spirv-cross-build}"
output="${IMB_SPIRV_CROSS_OUTPUT:-${repo_root}/build/tools/spirv-cross}"

for required_command in git cmake; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "build-spirv-cross: missing required command: ${required_command}" >&2
        exit 1
    fi
done

if [[ ! -d "${source_dir}/.git" ]]; then
    mkdir -p "$(dirname "${source_dir}")"
    git clone --filter=blob:none --no-checkout \
        https://github.com/KhronosGroup/SPIRV-Cross.git "${source_dir}"
fi

if [[ "$(git -C "${source_dir}" rev-parse HEAD 2>/dev/null || true)" != "${spirv_cross_commit}" ]]; then
    git -C "${source_dir}" fetch --depth 1 origin "${spirv_cross_commit}"
    git -C "${source_dir}" checkout --detach "${spirv_cross_commit}"
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPIRV_CROSS_CLI=ON \
    -DSPIRV_CROSS_ENABLE_TESTS=OFF \
    -DSPIRV_CROSS_ENABLE_C_API=OFF
cmake --build "${build_dir}" --config Release --target spirv-cross --parallel

mkdir -p "$(dirname "${output}")"
cp "${build_dir}/spirv-cross" "${output}"
chmod 0755 "${output}"
echo "build-spirv-cross: built ${output} from Khronos ${spirv_cross_commit}"
