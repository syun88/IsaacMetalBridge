#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
image="${1:-${IMB_ISAAC_BRIDGE_IMAGE:-imb-isaac-sim:6.0.1-dev}}"
cache_dir="${repo_root}/build/runtime/assets/robot-warehouse"
asset_root_url="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"

mkdir -p "${cache_dir}"
container run --rm \
    --name "imb-asset-cache-$$" \
    --platform linux/arm64 \
    --mount "type=bind,source=${cache_dir},target=/opt/imb-assets" \
    --mount "type=bind,source=${script_dir},target=/opt/imb-scripts,readonly" \
    --entrypoint /bin/sh \
    "${image}" \
    -c 'usd=/isaac-sim/extscache/omni.usd.libs-1.0.3+f9bf0dda.la64.r.cp312; PYTHONPATH="$usd" LD_LIBRARY_PATH="$usd/bin:/isaac-sim/kit/lib" exec /isaac-sim/kit/python/bin/python3 /opt/imb-scripts/cache-isaac-usd-tree.py "$@"' \
    imb-asset-cache \
    "${asset_root_url}" \
    /opt/imb-assets \
    Isaac/Environments/Simple_Warehouse/warehouse.usd \
    Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd

echo "cache-isaac-robot-warehouse: official NVIDIA assets ready at ${cache_dir}"
