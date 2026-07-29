#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
viewer_binary="${repo_root}/host/.build/release/imb-viewer"
viewer_app="${repo_root}/build/IsaacSimMetalViewer.app"

if [[ "${IMB_SKIP_VIEWER_BUILD:-0}" != "1" ]]; then
    swift build --package-path "${repo_root}/host" -c release --product imb-viewer
fi

mkdir -p "${viewer_app}/Contents/MacOS"
cp "${repo_root}/host/ViewerApp/Info.plist" "${viewer_app}/Contents/Info.plist"
cp "${viewer_binary}" "${viewer_app}/Contents/MacOS/IsaacSimMetalViewer"
chmod 0755 "${viewer_app}/Contents/MacOS/IsaacSimMetalViewer"
codesign --force --deep --sign - "${viewer_app}"

echo "build-viewer-app: built ${viewer_app}"
