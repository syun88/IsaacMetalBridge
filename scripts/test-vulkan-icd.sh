#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
adapter="${repo_root}/host/ContainerAdapter/.build/release/imb-container-host"
spirv_cross="${IMB_SPIRV_CROSS:-${repo_root}/build/tools/spirv-cross}"
image="imb-guest:dev"
container_id="imb-vulkan-probe-$$"
vsock_port="19002"
adapter_log="$(mktemp -t imb-vulkan-adapter.XXXXXX)"
guest_log="$(mktemp -t imb-vulkan-guest.XXXXXX)"
artifact_dir="${repo_root}/build/artifacts"
vulkan_ppm="${artifact_dir}/imb-vulkan-triangle.ppm"
vulkan_png="${artifact_dir}/imb-vulkan-triangle.png"
builder_was_running=0

"${script_dir}/cleanup-bridge-containers.sh"

if [[ "$(container builder status 2>/dev/null | awk 'NR == 2 {print $3}')" == "running" ]]; then
    builder_was_running=1
fi

cleanup() {
    exit_status=$?
    container delete --force "${container_id}" >/dev/null 2>&1 || true
    if [[ "${builder_was_running}" -eq 0 ]]; then
        container builder stop >/dev/null 2>&1 || true
    fi
    if [[ "${exit_status}" -eq 0 ]]; then
        find "${adapter_log}" "${guest_log}" -delete >/dev/null 2>&1 || true
    else
        echo "test-vulkan-icd: preserved failure logs adapter=${adapter_log} guest=${guest_log}" >&2
    fi
}
trap cleanup EXIT

if [[ ! -x "${adapter}" \
    || -n "$(find "${repo_root}/host/Sources" "${repo_root}/host/ContainerAdapter/Sources" \
        -type f -newer "${adapter}" -print -quit 2>/dev/null)" \
    || "${repo_root}/host/Package.swift" -nt "${adapter}" \
    || "${repo_root}/host/ContainerAdapter/Package.swift" -nt "${adapter}" ]]; then
    "${script_dir}/build-container-adapter.sh"
fi
if [[ ! -x "${spirv_cross}" ]]; then
    IMB_SPIRV_CROSS_OUTPUT="${spirv_cross}" "${script_dir}/build-spirv-cross.sh"
fi

if [[ "${IMB_SKIP_GUEST_BUILD:-0}" != "1" ]]; then
    container build \
        --platform linux/arm64 \
        --file "${repo_root}/guest/Containerfile" \
        --tag "${image}" \
        --progress plain \
        "${repo_root}"
fi
mkdir -p "${artifact_dir}"

container run \
    --detach \
    --name "${container_id}" \
    --platform linux/arm64 \
    --uid "$(id -u)" \
    --gid "$(id -g)" \
    --volume "${artifact_dir}:/output" \
    --env "IMB_VSOCK_PORT=${vsock_port}" \
    --env "IMB_VULKAN_SPIRV_COMPUTE=1" \
    --env "IMB_VULKAN_GENERIC_COMPUTE=1" \
    --env "IMB_VULKAN_SPARSE_IMAGES=1" \
    --env "IMB_VULKAN_COMPUTE_TRACE=${IMB_VULKAN_COMPUTE_TRACE:-}" \
    --env "VK_DRIVER_FILES=/usr/local/share/vulkan/icd.d/imb_icd.json" \
    --entrypoint /usr/local/bin/imb-vulkan-probe \
    "${image}" \
    /output/imb-vulkan-triangle.ppm >/dev/null

connected=0
for _ in {1..40}; do
    if IMB_SPIRV_CROSS="${spirv_cross}" "${adapter}" \
        --container "${container_id}" \
        --vsock-port "${vsock_port}" >"${adapter_log}" 2>&1; then
        connected=1
        break
    fi
    sleep 0.25
done

container logs "${container_id}" >"${guest_log}" 2>&1 || true

if [[ "${connected}" -ne 1 ]]; then
    echo "test-vulkan-icd: host could not complete the ICD vsock session" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

if ! grep -F 'VULKAN_ICD discovered="IsaacMetalBridge (' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan loader did not discover the IMB physical device" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

if ! grep -F 'VULKAN_COMPUTE input=[1,2,3,4] addend=5 output=[6,7,8,9] backend=Metal fence=signaled' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan compute did not complete through the Metal bridge" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_MEMORY_BUDGET usage=' "${guest_log}" >/dev/null \
    || ! grep -F 'live_allocations=passed' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan memory budget still reports zero live GPU usage" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

if ! grep -F 'VULKAN_RASTER triangle=64x64 format=RGBA8' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan raster did not complete through the Metal bridge" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_BLAS triangles=2 geometries=opaque backend=Metal fence=signaled' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan BLAS did not complete through the Metal bridge" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_AS_COMMAND_ORDER copies=2 builds=2 shared_input=1 preserved=passed' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: interleaved transfer/BLAS command order was not preserved" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_TLAS instances=2 child_blas=2 backend=Metal fence=signaled' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan TLAS did not complete through the Metal bridge" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_RT_PIPELINE stages=1 groups=1 recursion=3 linked_groups=1 accepted=1' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan depth-3 KHR ray pipeline library did not link" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

if ! grep -F 'VULKAN_RAY_DISPATCH rays=64x64 center=hit corner=miss backend=Metal fence=signaled' "${guest_log}" >/dev/null; then
    cat "${guest_log}" >&2
    echo "test-vulkan-icd: Vulkan ray dispatch did not intersect the real Metal TLAS" >&2
    exit 1
fi
if ! grep -F 'VULKAN_SPARSE_IMAGE format=RGBA8 tile=128x128 map=passed unmap=passed backend=Metal' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan sparse image tiles did not map and unmap through Metal" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_SPARSE_IMAGE format=RGBA8_SRGB tile=128x128 map=passed unmap=passed backend=Metal' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan RGBA8 sRGB sparse image tiles did not map and unmap through Metal" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
for sparse_format in BC3_SRGB BC5_UNORM; do
    if ! grep -E "VULKAN_SPARSE_IMAGE format=${sparse_format} tile=[1-9][0-9]*x[1-9][0-9]* map=passed upload=passed unmap=passed backend=Metal" "${guest_log}" >/dev/null; then
        echo "test-vulkan-icd: Vulkan ${sparse_format} sparse image data did not upload, map, and unmap through Metal" >&2
        sed -n '1,260p' "${guest_log}" >&2
        exit 1
    fi
done
if ! grep -F 'VULKAN_IMAGE_READBACK format=RGBA16 mips=4 bytes=680 buffer_to_image=passed image_to_buffer=passed' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan RGBA16 mip upload/readback did not pass" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_IMAGE_SNORM format=RGBA8_SNORM extent=2x2 signed_bytes=passed buffer_to_image=passed image_to_buffer=passed' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan RGBA8_SNORM signed-byte upload/readback did not pass" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if ! grep -F 'VULKAN_EXTERNAL_MEMORY image=RGBA8 handle=OPAQUE_FD roundtrip=passed' "${guest_log}" >/dev/null; then
    echo "test-vulkan-icd: Vulkan OPAQUE_FD image-memory compatibility did not pass" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi
if [[ ! -s "${vulkan_ppm}" ]]; then
    echo "test-vulkan-icd: Vulkan raster did not create ${vulkan_ppm}" >&2
    exit 1
fi
sips -s format png "${vulkan_ppm}" --out "${vulkan_png}" >/dev/null

sed -n '1,240p' "${adapter_log}"
sed -n '1,240p' "${guest_log}"

echo "test-vulkan-icd: Vulkan compute, raster, RGBA8_SNORM ordinary-image transfer, BLAS, TLAS, depth-3 KHR pipeline, Metal sparse-image residency, OPAQUE_FD, and real Metal ray dispatch validation passed; image=${vulkan_png}"
