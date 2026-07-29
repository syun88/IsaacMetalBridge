#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
host_binary="${repo_root}/host/.build/release/imb-host"
guest_binary="${repo_root}/build/guest/imb-guest-probe"

if [[ ! -x "${host_binary}" || ! -x "${guest_binary}" ]]; then
    echo "run-guest-probe: binaries are missing; run scripts/build-all.sh first" >&2
    exit 1
fi

"${guest_binary}" --host "${host_binary}"
