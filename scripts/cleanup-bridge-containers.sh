#!/usr/bin/env bash
set -euo pipefail

# Remove only containers owned by this project. Apple container's `buildkit`
# service and unrelated user containers are intentionally left untouched.
removed=0
while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] || continue
    case "${container_id}" in
        imb-*)
            echo "cleanup-bridge-containers: deleting stale ${container_id}"
            container delete --force "${container_id}" >/dev/null
            removed=$((removed + 1))
            ;;
    esac
done < <(container list --all --quiet)

echo "cleanup-bridge-containers: removed ${removed} stale bridge container(s)"
