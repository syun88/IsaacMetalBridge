#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"

swift build --package-path "${repo_root}/host" -c release
IMB_SKIP_VIEWER_BUILD=1 "${script_dir}/build-viewer-app.sh"
cmake -S "${repo_root}/guest" -B "${repo_root}/build/guest" -DCMAKE_BUILD_TYPE=Release
cmake --build "${repo_root}/build/guest" --config Release

echo "build-all: built Swift host, native viewer app, and C++20 guest"
