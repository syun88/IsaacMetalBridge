#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
output_dir="${IMB_VULKAN_OVERRIDE_OUTPUT:-${repo_root}/build/vulkan-override}"
builder_image="${IMB_VULKAN_BUILDER_IMAGE:-imb-bridge-build:dev}"
container_id="imb-vulkan-override-build-$$"

cleanup() {
    container delete --force "${container_id}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${output_dir}"
container run \
    --name "${container_id}" \
    --platform linux/arm64 \
    --memory 2g \
    --uid "$(id -u)" \
    --gid "$(id -g)" \
    --volume "${repo_root}:/work:ro" \
    --volume "${output_dir}:/output" \
    --entrypoint /bin/sh \
    "${builder_image}" \
    -lc 'c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
        -fPIC -fvisibility=hidden -shared \
        -I/work/protocol/include -I/work/guest/src -I/src/vulkan \
        /work/guest/vulkan_icd/imb_vulkan_icd.cpp \
        -o /output/libimb_vulkan_icd.so \
    && glslangValidator -V \
        /work/guest/vulkan_icd/texel_add.comp \
        -o /tmp/imb_texel_add.spv \
    && xxd -i -n imb_texel_add_spv \
        /tmp/imb_texel_add.spv \
        /tmp/texel_add_spv.h \
    && c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -I/tmp \
        /work/guest/vulkan_icd/imb_vulkan_probe.cpp \
        -lvulkan \
        -o /output/imb-vulkan-probe \
    && c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
        -fPIC -fvisibility=hidden -shared \
        /work/guest/cuda_shim/imb_cuda_shim.cpp \
        -Wl,-soname,libcuda.so.1 \
        -o /output/libcuda.so.1 \
    && c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
        -fPIC -shared \
        /work/guest/timer_shim/imb_clock_shim.cpp \
        -ldl \
        -Wl,-soname,libimb_clock_shim.so \
        -o /output/libimb_clock_shim.so \
    && ln -sf libcuda.so.1 /output/libcuda.so \
    && c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
        /work/guest/cuda_shim/imb_cuda_external_probe.cpp \
        -L/output -Wl,-rpath,/output -lcuda \
        -o /tmp/imb-cuda-external-probe \
    && /tmp/imb-cuda-external-probe'

# The production image manifest intentionally points at /usr/local/lib.  A
# bind-mounted override needs its own absolute path or the loader will silently
# keep using the image-baked ICD instead of the newly compiled library.
cp "${repo_root}/guest/vulkan_icd/imb_icd_override.json" "${output_dir}/imb_icd.json"
chmod 0755 "${output_dir}/libcuda.so.1"
chmod 0755 "${output_dir}/imb-vulkan-probe"
chmod 0644 \
    "${output_dir}/libimb_vulkan_icd.so" \
    "${output_dir}/libimb_clock_shim.so" \
    "${output_dir}/imb_icd.json"
echo "build-isaac-vulkan-override: built Vulkan ICD and validated CUDA OPAQUE_FD mipmap interop in ${output_dir}"
