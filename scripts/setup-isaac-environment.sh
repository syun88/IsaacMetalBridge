#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
configured=0
failed=0

# Read only known keys from the ignored local file. Do not source arbitrary shell code.
local_env_file="${repo_root}/.env"
if [[ -f "${local_env_file}" ]]; then
    while IFS='=' read -r config_key config_value; do
        config_value="${config_value%$'\r'}"
        case "${config_key}" in
            IMB_ISAAC_IMAGE)
                if [[ -z "${IMB_ISAAC_IMAGE:-}" ]]; then
                    IMB_ISAAC_IMAGE="${config_value}"
                fi
                ;;
            IMB_ISAAC_PLATFORM)
                if [[ -z "${IMB_ISAAC_PLATFORM:-}" ]]; then
                    IMB_ISAAC_PLATFORM="${config_value}"
                fi
                ;;
            ISAAC_SIM_PATH)
                if [[ -z "${ISAAC_SIM_PATH:-}" ]]; then
                    ISAAC_SIM_PATH="${config_value}"
                fi
                ;;
        esac
    done < "${local_env_file}"
fi

if command -v container >/dev/null 2>&1; then
    echo "[ok] Apple container CLI: $(command -v container)"
else
    echo "[blocked] Apple container CLI is not installed; Phase 0 cannot run" >&2
    failed=1
fi

if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
    if [[ ! -d "${ISAAC_SIM_PATH}" ]]; then
        echo "[blocked] ISAAC_SIM_PATH is not a directory: ${ISAAC_SIM_PATH}" >&2
        failed=1
    else
        isaac_path="$(cd "${ISAAC_SIM_PATH}" && pwd -P)"
        case "${isaac_path}/" in
            "${repo_root}/"*)
                echo "[blocked] Isaac Sim must be stored outside this repository" >&2
                failed=1
                ;;
            *)
                configured=1
                echo "[ok] external Isaac Sim directory: ${isaac_path}"
                for required_file in isaac-sim.sh python.sh kit/kit; do
                    if [[ -e "${isaac_path}/${required_file}" ]]; then
                        echo "[ok] found ${required_file}"
                    else
                        echo "[blocked] missing expected real Isaac Sim file: ${required_file}" >&2
                        failed=1
                    fi
                done
                if [[ -e "${isaac_path}/kit/kit" ]]; then
                    kit_description="$(file "${isaac_path}/kit/kit")"
                    echo "[info] ${kit_description}"
                    if [[ "${kit_description}" != *"ARM aarch64"* && "${kit_description}" != *"arm64"* ]]; then
                        echo "[blocked] Kit executable is not identified as Linux ARM64/aarch64" >&2
                        failed=1
                    fi
                fi
                ;;
        esac
    fi
fi

if [[ -n "${IMB_ISAAC_IMAGE:-}" ]]; then
    configured=1
    echo "[configured] external image reference: ${IMB_ISAAC_IMAGE}"
    if command -v container >/dev/null 2>&1; then
        if image_json="$(container image inspect "${IMB_ISAAC_IMAGE}" 2>/dev/null)"; then
            echo "[ok] image is present in the Apple container image store"
            if command -v jq >/dev/null 2>&1; then
                expected_platform="${IMB_ISAAC_PLATFORM:-linux/arm64}"
                actual_platform="$(printf '%s' "${image_json}" | jq -r '.[0].variants[0].platform | "\(.os)/\(.architecture)"')"
                manifest_digest="$(printf '%s' "${image_json}" | jq -r '.[0].variants[0].digest')"
                if [[ "${actual_platform}" != "${expected_platform}" ]]; then
                    echo "[blocked] image platform is ${actual_platform}, expected ${expected_platform}" >&2
                    failed=1
                else
                    echo "[ok] image platform: ${actual_platform}"
                fi
                echo "[ok] image manifest: ${manifest_digest}"
            else
                echo "[info] jq is unavailable; image architecture was not decoded"
            fi
        else
            echo "[blocked] image is not loaded in the Apple container image store" >&2
            failed=1
        fi
    fi
    echo "[info] this script will not pull the image, authenticate, accept NVIDIA terms, or launch Isaac Sim"
fi

if [[ ${configured} -eq 0 ]]; then
    echo "[blocked] set ISAAC_SIM_PATH or IMB_ISAAC_IMAGE for the real Isaac Sim 6.0.1 ARM64 artifact" >&2
    failed=1
fi

if [[ ${failed} -ne 0 ]]; then
    echo "setup-isaac-environment: prerequisites incomplete; no package was downloaded and no simulator was started" >&2
    exit 1
fi

echo "setup-isaac-environment: configuration validated"
echo "setup-isaac-environment: after accepting NVIDIA's EULA, run: ACCEPT_EULA=Y ./scripts/run-isaac-sim.sh"
