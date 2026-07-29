#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
adapter="${repo_root}/host/ContainerAdapter/.build/release/imb-container-host"
image="imb-guest:dev"
container_id="imb-vsock-probe-$$"
vsock_port="19001"
adapter_log="$(mktemp -t imb-container-adapter.XXXXXX)"
artifact_dir="${repo_root}/build/artifacts"
container_ppm="${artifact_dir}/imb-metal-triangle-container.ppm"
container_png="${artifact_dir}/imb-metal-triangle-container.png"
builder_was_running=0

if [[ "$(container builder status 2>/dev/null | awk 'NR == 2 {print $3}')" == "running" ]]; then
    builder_was_running=1
fi

cleanup() {
    container delete --force "${container_id}" >/dev/null 2>&1 || true
    if [[ "${builder_was_running}" -eq 0 ]]; then
        container builder stop >/dev/null 2>&1 || true
    fi
    find "${adapter_log}" -delete >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${script_dir}/build-container-adapter.sh"
mkdir -p "${artifact_dir}"

container build \
    --platform linux/arm64 \
    --file "${repo_root}/guest/Containerfile" \
    --tag "${image}" \
    --progress plain \
    "${repo_root}"

container run \
    --detach \
    --name "${container_id}" \
    --platform linux/arm64 \
    --uid "$(id -u)" \
    --gid "$(id -g)" \
    --volume "${artifact_dir}:/output" \
    "${image}" \
    --vsock-listen "${vsock_port}" \
    --image-output /output/imb-metal-triangle-container.ppm >/dev/null

connected=0
for _ in {1..40}; do
    if "${adapter}" \
        --container "${container_id}" \
        --vsock-port "${vsock_port}" >"${adapter_log}" 2>&1; then
        connected=1
        break
    fi
    sleep 0.25
done

if [[ "${connected}" -ne 1 ]]; then
    echo "test-container-vsock: host could not dial guest vsock listener" >&2
    sed -n '1,240p' "${adapter_log}" >&2
    container logs "${container_id}" >&2 || true
    exit 1
fi

sed -n '1,240p' "${adapter_log}"
guest_log="$(container logs "${container_id}")"
echo "${guest_log}"

if ! grep -Fq "METAL_RASTER DRAW_TRIANGLE verified" <<<"${guest_log}"; then
    echo "test-container-vsock: guest did not verify Metal raster output" >&2
    exit 1
fi
if [[ ! -s "${container_ppm}" ]]; then
    echo "test-container-vsock: guest did not create ${container_ppm}" >&2
    exit 1
fi
sips -s format png "${container_ppm}" --out "${container_png}" >/dev/null
if [[ ! -s "${container_png}" ]]; then
    echo "test-container-vsock: PNG conversion did not create ${container_png}" >&2
    exit 1
fi

echo "test-container-vsock: real Apple container/vsock/Metal raster passed; image=${container_png}"
