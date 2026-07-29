#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
strict=0
issues=0
configured_isaac_image="${IMB_ISAAC_IMAGE:-}"

if [[ -z "${configured_isaac_image}" && -f "${repo_root}/.env" ]]; then
    configured_isaac_image="$(awk -F= '$1 == "IMB_ISAAC_IMAGE" {sub(/^[^=]*=/, ""); print; exit}' "${repo_root}/.env")"
fi

if [[ "${1:-}" == "--strict" ]]; then
    strict=1
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--strict]" >&2
    exit 2
fi

ok() { echo "[ok]      $*"; }
warning() { echo "[warning] $*"; issues=$((issues + 1)); }

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    ok "Apple Silicon macOS $(sw_vers -productVersion)"
else
    warning "development host is not Apple Silicon macOS"
fi

for required_command in git swift cmake c++; do
    if command -v "${required_command}" >/dev/null 2>&1; then
        ok "${required_command}: $(command -v "${required_command}")"
    else
        warning "missing build command: ${required_command}"
    fi
done

if command -v container >/dev/null 2>&1; then
    ok "Apple container CLI: $(command -v container)"
else
    warning "Apple container CLI is not installed (Phase 0 blocked)"
fi

if [[ -d "${repo_root}/third_party/apple-container/.git" || -f "${repo_root}/third_party/apple-container/.git" ]]; then
    apple_commit="$(git -C "${repo_root}/third_party/apple-container" rev-parse HEAD)"
    if [[ "${apple_commit}" == "14233cee65486c1ada2b82403c17d1236a9176c2" ]]; then
        ok "Apple container pinned at ${apple_commit}"
    else
        warning "Apple container is at unexpected commit ${apple_commit}"
    fi
    if [[ -z "$(git -C "${repo_root}/third_party/apple-container" status --short)" ]]; then
        ok "Apple container submodule is clean"
    else
        warning "Apple container submodule has tracked modifications"
    fi
else
    warning "Apple container submodule is not initialized"
fi

if [[ -n "${ISAAC_SIM_PATH:-}" ]]; then
    ok "ISAAC_SIM_PATH is configured: ${ISAAC_SIM_PATH}"
elif [[ -n "${configured_isaac_image}" ]]; then
    if command -v container >/dev/null 2>&1 && container image inspect "${configured_isaac_image}" >/dev/null 2>&1; then
        ok "real Isaac Sim image is loaded: ${configured_isaac_image}"
    else
        warning "IMB_ISAAC_IMAGE is configured but not loaded: ${configured_isaac_image}"
    fi
else
    warning "no external real Isaac Sim path or image is configured (Phase 2 blocked)"
fi

if [[ ${issues} -eq 0 ]]; then
    echo "doctor: all build and integration prerequisites are present"
elif [[ ${strict} -eq 1 ]]; then
    echo "doctor: ${issues} issue(s) found" >&2
    exit 1
else
    echo "doctor: ${issues} integration warning(s); use --strict to make warnings fatal"
fi
