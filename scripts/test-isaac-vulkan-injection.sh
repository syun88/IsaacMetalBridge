#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
adapter="${repo_root}/host/ContainerAdapter/.build/release/imb-container-host"
source_image="${IMB_ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
derived_image="${IMB_ISAAC_BRIDGE_IMAGE:-imb-isaac-sim:6.0.1-dev}"
vulkan_override="${IMB_VULKAN_ICD_OVERRIDE:-${repo_root}/build/vulkan-override}"
spirv_cross="${IMB_SPIRV_CROSS:-${repo_root}/build/tools/spirv-cross}"
container_id="imb-isaac-vulkan-probe-$$"
vsock_port="19003"
adapter_log="$(mktemp -t imb-isaac-adapter.XXXXXX)"
guest_log="$(mktemp -t imb-isaac-guest.XXXXXX)"
builder_was_running=0

if [[ "$(container builder status 2>/dev/null | awk 'NR == 2 {print $3}')" == "running" ]]; then
    builder_was_running=1
fi

cleanup() {
    container delete --force "${container_id}" >/dev/null 2>&1 || true
    if [[ "${builder_was_running}" -eq 0 ]]; then
        container builder stop >/dev/null 2>&1 || true
    fi
    find "${adapter_log}" "${guest_log}" -delete >/dev/null 2>&1 || true
}
trap cleanup EXIT

if ! container image inspect "${source_image}" >/dev/null 2>&1; then
    echo "test-isaac-vulkan-injection: real Isaac Sim image is not loaded: ${source_image}" >&2
    exit 1
fi

if [[ ! -x "${adapter}" ]]; then
    "${script_dir}/build-container-adapter.sh"
fi

if [[ "${IMB_SKIP_ISAAC_BRIDGE_BUILD:-0}" != "1" ]]; then
    container build \
        --platform linux/arm64 \
        --build-arg "ISAAC_IMAGE=${source_image}" \
        --file "${repo_root}/guest/IsaacContainerfile" \
        --tag "${derived_image}" \
        --progress plain \
        "${repo_root}"
fi

container run \
    --detach \
    --name "${container_id}" \
    --platform linux/arm64 \
    --env "IMB_VSOCK_PORT=${vsock_port}" \
    --env "IMB_VULKAN_NVIDIA_COMPAT=0" \
    --env "IMB_VULKAN_TRACE=1" \
    --env "IMB_VULKAN_SPIRV_COMPUTE=1" \
    --env "IMB_VULKAN_GENERIC_COMPUTE=1" \
    --env "IMB_VULKAN_SPARSE_IMAGES=1" \
    --env "VK_DRIVER_FILES=/opt/imb-override/imb_icd.json" \
    --volume "${vulkan_override}:/opt/imb-override:ro" \
    --entrypoint /opt/imb-override/imb-vulkan-probe \
    "${derived_image}" >/dev/null

connected=0
for _ in {1..80}; do
    if IMB_SPIRV_CROSS="${spirv_cross}" "${adapter}" \
        --container "${container_id}" \
        --vsock-port "${vsock_port}" >"${adapter_log}" 2>&1; then
        connected=1
        break
    else
        adapter_attempt_status=$?
        if [[ "${adapter_attempt_status}" -eq 133 || "${adapter_attempt_status}" -eq 134 ]]; then
            break
        fi
    fi
    sleep 0.25
done

container logs "${container_id}" >"${guest_log}" 2>&1 || true

if [[ "${connected}" -ne 1 ]]; then
    echo "test-isaac-vulkan-injection: host could not complete the Isaac-image ICD session" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

if ! grep -F 'VULKAN_ICD discovered="IsaacMetalBridge (' "${guest_log}" >/dev/null \
    || ! grep -F 'VULKAN_COMPUTE input=[1,2,3,4] addend=5 output=[6,7,8,9] backend=Metal fence=signaled' "${guest_log}" >/dev/null \
    || ! grep -F 'VULKAN_TEXEL_BUFFER format=R32_UINT output=[8,9,10,11] backend=Metal fence=signaled' "${guest_log}" >/dev/null \
    || ! grep -F 'executed Metal compute pipeline=' "${guest_log}" >/dev/null \
    || ! grep -F 'VULKAN_RASTER triangle=64x64 format=RGBA8' "${guest_log}" >/dev/null; then
    echo "test-isaac-vulkan-injection: injected ICD did not complete Metal compute, texel-buffer, and raster" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    sed -n '1,240p' "${guest_log}" >&2
    exit 1
fi

sed -n '1,40p' "${adapter_log}"
grep -E '^(VULKAN_|imb-vulkan-icd: executed Metal compute pipeline=)' "${guest_log}"
echo "test-isaac-vulkan-injection: real Isaac Sim ARM64 image completed Metal compute, texel-buffer, and raster"
