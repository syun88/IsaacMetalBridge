#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"
adapter="${repo_root}/host/ContainerAdapter/.build/release/imb-container-host"
viewer_app="${repo_root}/build/IsaacSimMetalViewer.app"
viewer="${viewer_app}/Contents/MacOS/IsaacSimMetalViewer"
source_image="${IMB_ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
derived_image="${IMB_ISAAC_BRIDGE_IMAGE:-imb-isaac-sim:6.0.1-dev}"
simple_grid_url="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Grid/default_environment.usd"
simple_grid_texture_base_url="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Environments/Grid/Materials/Textures"
simple_grid_texture_names=(
    "Wireframe_blue.png"
    "WireframeBlur_blue.png"
    "WireframeBlur_basecolor.png"
)
simple_grid_cache=""
startup_stage=""
vsock_port="${IMB_VSOCK_PORT:-19004}"
container_id="${IMB_ISAAC_CONTAINER_NAME:-imb-isaac-sim-$$}"
experience="base"
quit_after=""
keep_container="${IMB_KEEP_ISAAC_CONTAINER:-0}"
skip_build="${IMB_SKIP_ISAAC_BRIDGE_BUILD:-0}"
adapter_sessions="${IMB_ADAPTER_SESSIONS:-1}"
spirv_compute="${IMB_VULKAN_SPIRV_COMPUTE:-1}"
generic_compute="${IMB_VULKAN_GENERIC_COMPUTE:-1}"
sparse_images="${IMB_VULKAN_SPARSE_IMAGES:-1}"
spirv_cross="${IMB_SPIRV_CROSS:-${repo_root}/build/tools/spirv-cross}"
show_window=0
demo_scene=0
animate_demo=0
simple_grid=0
enable_ros2=0
viewer_pid=""
frame_output="${IMB_FRAME_OUTPUT:-}"
input_output=""
runtime_dir="${repo_root}/build/runtime"
camera_dir=""
camera_state_output=""
camera_sensor_output=""
camera_sensor_host_dir=""
camera_sensor_name=""
camera_sensor_frame_output=""
camera_sensor_staged_output=""
physics_smoke_output=""
physics_smoke_host_dir=""
physics_smoke_name=""
physics_smoke_staged_output=""
scene_material_texture=""
ld_preload_value=""
extra_args=()
log_pid=""
sensor_copy_pid=""
physics_copy_pid=""
builder_was_running=0

usage() {
    cat <<'USAGE'
usage: run-isaac-sim.sh [--experience base|full] [--quit-after FRAMES]
                        [--window] [--demo-scene [--animate-demo]|--simple-grid]
                        [--camera-sensor-output FILE]
                        [--physics-smoke-output FILE]
                        [--ros2] [--keep-container] [--no-build]
                        [-- KIT_ARGS...]

Kit runs inside the Linux VM and --window displays its real UI frames in a
native macOS window. Both the base and full Isaac Sim experiences are
available. Full startup uses a targeted ARM virtual-counter compatibility shim.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experience)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            experience="$2"
            shift 2
            ;;
        --quit-after)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+([.][0-9]+)?$ ]] || { usage >&2; exit 2; }
            quit_after="$2"
            shift 2
            ;;
        --keep-container)
            keep_container=1
            shift
            ;;
        --no-build)
            skip_build=1
            shift
            ;;
        --window)
            show_window=1
            shift
            ;;
        --demo-scene)
            demo_scene=1
            shift
            ;;
        --animate-demo)
            animate_demo=1
            shift
            ;;
        --simple-grid)
            simple_grid=1
            shift
            ;;
        --camera-sensor-output)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            camera_sensor_output="$2"
            shift 2
            ;;
        --physics-smoke-output)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            physics_smoke_output="$2"
            shift 2
            ;;
        --ros2)
            enable_ros2=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${experience}" != "base" && "${experience}" != "full" ]]; then
    echo "run-isaac-sim: experience must be base or full" >&2
    exit 2
fi
if [[ "${demo_scene}" -eq 1 && "${simple_grid}" -eq 1 ]]; then
    echo "run-isaac-sim: --demo-scene and --simple-grid are mutually exclusive" >&2
    exit 2
fi
if [[ "${animate_demo}" -eq 1 && "${demo_scene}" -ne 1 ]]; then
    echo "run-isaac-sim: --animate-demo requires --demo-scene" >&2
    exit 2
fi
if [[ -n "${camera_sensor_output}" \
    && "${demo_scene}" -ne 1 && "${simple_grid}" -ne 1 ]]; then
    echo "run-isaac-sim: --camera-sensor-output requires --demo-scene or --simple-grid" >&2
    exit 2
fi
if [[ -n "${physics_smoke_output}" && "${demo_scene}" -ne 1 ]]; then
    echo "run-isaac-sim: --physics-smoke-output requires --demo-scene" >&2
    exit 2
fi
if [[ "${animate_demo}" -eq 1 && -n "${physics_smoke_output}" ]]; then
    echo "run-isaac-sim: --animate-demo and --physics-smoke-output are mutually exclusive" >&2
    exit 2
fi
camera_sensor_width="${IMB_CAMERA_SENSOR_WIDTH:-640}"
camera_sensor_height="${IMB_CAMERA_SENSOR_HEIGHT:-480}"
if [[ ! "${camera_sensor_width}" =~ ^[0-9]+$ \
    || "${camera_sensor_width}" -lt 16 || "${camera_sensor_width}" -gt 8192 \
    || ! "${camera_sensor_height}" =~ ^[0-9]+$ \
    || "${camera_sensor_height}" -lt 16 || "${camera_sensor_height}" -gt 8192 ]]; then
    echo "run-isaac-sim: IMB_CAMERA_SENSOR_WIDTH/HEIGHT must be integers in 16..8192" >&2
    exit 2
fi
if [[ "${enable_ros2}" -eq 1 && "${experience}" != "full" ]]; then
    echo "run-isaac-sim: --ros2 currently requires --experience full" >&2
    exit 2
fi
if [[ ! "${adapter_sessions}" =~ ^[1-9][0-9]*$ ]]; then
    echo "run-isaac-sim: IMB_ADAPTER_SESSIONS must be a positive integer" >&2
    exit 2
fi
if [[ "${sparse_images}" != "0" && "${sparse_images}" != "1" ]]; then
    echo "run-isaac-sim: IMB_VULKAN_SPARSE_IMAGES must be 0 or 1" >&2
    exit 2
fi
if [[ "${generic_compute}" != "0" && "${generic_compute}" != "1" ]]; then
    echo "run-isaac-sim: IMB_VULKAN_GENERIC_COMPUTE must be 0 or 1" >&2
    exit 2
fi

if [[ "${ACCEPT_EULA:-}" != "Y" ]]; then
    echo "run-isaac-sim: set ACCEPT_EULA=Y only after reviewing and accepting NVIDIA's Isaac Sim EULA" >&2
    exit 2
fi

required_commands=(container grep)
if [[ "${simple_grid}" -eq 1 ]]; then
    required_commands+=(curl od tr)
fi
for required_command in "${required_commands[@]}"; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "run-isaac-sim: missing required command: ${required_command}" >&2
        exit 1
    fi
done

if [[ "$(container builder status 2>/dev/null | awk 'NR == 2 {print $3}')" == "running" ]]; then
    builder_was_running=1
fi

if ! container image inspect "${source_image}" >/dev/null 2>&1; then
    echo "run-isaac-sim: real Isaac Sim image is not loaded: ${source_image}" >&2
    exit 1
fi

if [[ ! -x "${adapter}" \
    || -n "$(find "${repo_root}/host/Sources" "${repo_root}/host/ContainerAdapter/Sources" \
        -type f -newer "${adapter}" -print -quit 2>/dev/null)" \
    || "${repo_root}/host/Package.swift" -nt "${adapter}" \
    || "${repo_root}/host/ContainerAdapter/Package.swift" -nt "${adapter}" ]]; then
    "${script_dir}/build-container-adapter.sh"
fi

if [[ "${spirv_compute}" != "0" ]]; then
    if [[ ! -x "${spirv_cross}" ]]; then
        IMB_SPIRV_CROSS_OUTPUT="${spirv_cross}" "${script_dir}/build-spirv-cross.sh"
    fi
    export IMB_SPIRV_CROSS="${spirv_cross}"
fi

local_vulkan_override="${repo_root}/build/vulkan-override"
if [[ -z "${IMB_VULKAN_ICD_OVERRIDE+x}" ]]; then
    override_needs_build=0
    if [[ ! -f "${local_vulkan_override}/libimb_vulkan_icd.so" \
        || ! -f "${local_vulkan_override}/imb_icd.json" \
        || ! -f "${local_vulkan_override}/libcuda.so.1" ]]; then
        override_needs_build=1
    elif [[ -n "$(find \
        "${repo_root}/protocol/include" \
        "${repo_root}/guest/vulkan_icd" \
        "${repo_root}/guest/cuda_shim" \
        -type f -newer "${local_vulkan_override}/libimb_vulkan_icd.so" \
        -print -quit 2>/dev/null)" ]]; then
        override_needs_build=1
    fi
    if [[ "${override_needs_build}" -eq 1 ]]; then
        echo "run-isaac-sim: rebuilding the small local Vulkan/CUDA override"
        "${script_dir}/build-isaac-vulkan-override.sh"
    fi
    IMB_VULKAN_ICD_OVERRIDE="${local_vulkan_override}"
    echo "run-isaac-sim: using current local Vulkan/CUDA override ${IMB_VULKAN_ICD_OVERRIDE}"
fi

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "${log_pid}" ]]; then
        kill "${log_pid}" >/dev/null 2>&1 || true
        wait "${log_pid}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${viewer_pid}" ]]; then
        kill "${viewer_pid}" >/dev/null 2>&1 || true
        wait "${viewer_pid}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${sensor_copy_pid}" ]]; then
        kill "${sensor_copy_pid}" >/dev/null 2>&1 || true
        wait "${sensor_copy_pid}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${physics_copy_pid}" ]]; then
        kill "${physics_copy_pid}" >/dev/null 2>&1 || true
        wait "${physics_copy_pid}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${camera_sensor_output}" ]]; then
        rm -f \
            "${camera_sensor_output}.tmp.${container_id}" \
            "${camera_sensor_output}.json.tmp.${container_id}"
    fi
    if [[ -n "${physics_smoke_output}" ]]; then
        rm -f "${physics_smoke_output}.tmp.${container_id}"
    fi
    if [[ -n "${input_output}" ]]; then
        rm -f "${input_output}"
    fi
    if [[ "${keep_container}" != "1" ]]; then
        container delete --force "${container_id}" >/dev/null 2>&1 || true
    fi
    if [[ "${builder_was_running}" -eq 0 ]]; then
        container builder stop >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ "${skip_build}" != "1" ]]; then
    echo "run-isaac-sim: building ${derived_image} from the user-managed real image"
    container build \
        --platform linux/arm64 \
        --build-arg "ISAAC_IMAGE=${source_image}" \
        --file "${repo_root}/guest/IsaacContainerfile" \
        --tag "${derived_image}" \
        --progress plain \
        "${repo_root}"
elif ! container image inspect "${derived_image}" >/dev/null 2>&1; then
    echo "run-isaac-sim: --no-build was used but the bridge image is not loaded: ${derived_image}" >&2
    exit 1
fi

mkdir -p "${runtime_dir}"
if [[ "${demo_scene}" -eq 1 || "${simple_grid}" -eq 1 ]]; then
    camera_dir="${runtime_dir}/${container_id}-camera"
    camera_state_output="${camera_dir}/state.bin"
    mkdir -p "${camera_dir}"
    rm -f "${camera_state_output}"
fi
if [[ -n "${camera_sensor_output}" ]]; then
    mkdir -p "$(dirname "${camera_sensor_output}")"
    camera_sensor_host_dir="$(
        cd "$(dirname "${camera_sensor_output}")" && pwd -P
    )"
    camera_sensor_name="$(basename "${camera_sensor_output}")"
    if [[ -z "${camera_sensor_name}" || "${camera_sensor_name}" == "." \
        || "${camera_sensor_name}" == ".." ]]; then
        echo "run-isaac-sim: invalid --camera-sensor-output file name" >&2
        exit 2
    fi
    camera_sensor_output="${camera_sensor_host_dir}/${camera_sensor_name}"
    camera_sensor_frame_output="${camera_dir}/sensor-frame.ppm"
    camera_sensor_staged_output="${camera_dir}/sensor-output.ppm"
    rm -f "${camera_sensor_output}" "${camera_sensor_output}.json"
    rm -f \
        "${camera_sensor_frame_output}" \
        "${camera_sensor_staged_output}" \
        "${camera_sensor_staged_output}.json"
fi
if [[ -n "${physics_smoke_output}" ]]; then
    mkdir -p "$(dirname "${physics_smoke_output}")"
    physics_smoke_host_dir="$(
        cd "$(dirname "${physics_smoke_output}")" && pwd -P
    )"
    physics_smoke_name="$(basename "${physics_smoke_output}")"
    if [[ -z "${physics_smoke_name}" || "${physics_smoke_name}" == "." \
        || "${physics_smoke_name}" == ".." ]]; then
        echo "run-isaac-sim: invalid --physics-smoke-output file name" >&2
        exit 2
    fi
    physics_smoke_output="${physics_smoke_host_dir}/${physics_smoke_name}"
    physics_smoke_staged_output="${camera_dir}/physics-smoke.json"
    rm -f "${physics_smoke_output}" "${physics_smoke_staged_output}"
fi

if [[ "${show_window}" -eq 1 ]]; then
    if [[ ! -x "${viewer}" ]]; then
        "${script_dir}/build-viewer-app.sh"
    fi
    if [[ -z "${frame_output}" ]]; then
        frame_output="${runtime_dir}/${container_id}.ppm"
    fi
    input_output="${runtime_dir}/${container_id}.input"
    "${viewer}" \
        --frame "${frame_output}" \
        --input "${input_output}" \
        --title "Isaac Sim 6.0.1 — Apple Metal Bridge" \
        >"${runtime_dir}/${container_id}-viewer.log" 2>&1 &
    viewer_pid=$!
    echo "run-isaac-sim: opening native macOS viewer for ${frame_output}"
fi

if [[ "${simple_grid}" -eq 1 ]]; then
    simple_grid_cache="${repo_root}/build/runtime/assets/simple-grid/default_environment.usd"
    mkdir -p "$(dirname "${simple_grid_cache}")"
    if [[ ! -s "${simple_grid_cache}" \
        || "$(LC_ALL=C head -c 8 "${simple_grid_cache}" 2>/dev/null || true)" != "PXR-USDC" ]]; then
        simple_grid_partial="${simple_grid_cache}.part.$$"
        echo "run-isaac-sim: downloading explicitly requested NVIDIA Simple Grid stage"
        if ! curl --fail --location --silent --show-error \
            --output "${simple_grid_partial}" "${simple_grid_url}"; then
            rm -f "${simple_grid_partial}"
            echo "run-isaac-sim: failed to download NVIDIA Simple Grid stage" >&2
            exit 1
        fi
        if [[ "$(LC_ALL=C head -c 8 "${simple_grid_partial}" 2>/dev/null || true)" != "PXR-USDC" ]]; then
            rm -f "${simple_grid_partial}"
            echo "run-isaac-sim: downloaded Simple Grid stage is not a USDC file" >&2
            exit 1
        fi
        mv "${simple_grid_partial}" "${simple_grid_cache}"
    fi
    simple_grid_texture_directory="${repo_root}/build/runtime/assets/simple-grid/Materials/Textures"
    mkdir -p "${simple_grid_texture_directory}"
    for simple_grid_texture_name in "${simple_grid_texture_names[@]}"; do
        simple_grid_texture_cache="${simple_grid_texture_directory}/${simple_grid_texture_name}"
        simple_grid_texture_signature="$(
            od -An -tx1 -N8 "${simple_grid_texture_cache}" 2>/dev/null | tr -d ' \n' || true
        )"
        if [[ ! -s "${simple_grid_texture_cache}" \
            || "${simple_grid_texture_signature}" != "89504e470d0a1a0a" ]]; then
            simple_grid_texture_partial="${simple_grid_texture_cache}.part.$$"
            echo "run-isaac-sim: downloading NVIDIA Simple Grid texture ${simple_grid_texture_name}"
            if ! curl --fail --location --silent --show-error \
                --output "${simple_grid_texture_partial}" \
                "${simple_grid_texture_base_url}/${simple_grid_texture_name}"; then
                rm -f "${simple_grid_texture_partial}"
                echo "run-isaac-sim: failed to download NVIDIA Simple Grid texture ${simple_grid_texture_name}" >&2
                exit 1
            fi
            simple_grid_texture_signature="$(
                od -An -tx1 -N8 "${simple_grid_texture_partial}" 2>/dev/null | tr -d ' \n' || true
            )"
            if [[ "${simple_grid_texture_signature}" != "89504e470d0a1a0a" ]]; then
                rm -f "${simple_grid_texture_partial}"
                echo "run-isaac-sim: downloaded ${simple_grid_texture_name} is not a PNG file" >&2
                exit 1
            fi
            mv "${simple_grid_texture_partial}" "${simple_grid_texture_cache}"
        fi
    done
    scene_material_texture="${simple_grid_texture_directory}/Wireframe_blue.png"
fi

entrypoint="/isaac-sim/kit/kit"
kit_args=("/isaac-sim/apps/isaacsim.exp.base.kit" "--no-window")
if [[ "${experience}" == "full" ]]; then
    entrypoint="/isaac-sim/isaac-sim.sh"
    kit_args=("--no-window")
    if [[ "${enable_ros2}" -eq 0 ]]; then
        # NVIDIA's bundled Jazzy directory contains global copies of libraries
        # such as spdlog/crypto. Exposing the whole directory during early Full
        # startup currently trips a Carbonite TaskGroup ABI assertion on ARM.
        # Keep the stable Full path isolated unless ROS2 is explicitly tested,
        # and clear Full's Linux default bridge setting so app.setup does not
        # briefly load a bridge that cannot resolve libament_index_cpp.so.
        kit_args+=(
            "--no-ros-env"
            "--/isaac/startup/ros_bridge_extension="
            "--/isaac/startup/ros_sim_control_extension=false"
        )
    else
        echo "run-isaac-sim: enabling NVIDIA bundled ROS 2 Jazzy/FastDDS environment (experimental)"
    fi
fi
if [[ -n "${quit_after}" ]]; then
    kit_args+=(
        "--/app/quitAfter=${quit_after}"
        "--/app/fastShutdown=1"
        "--/app/file/ignoreUnsavedStage=1"
    )
fi
if [[ "${experience}" == "full" ]]; then
    # Carbonite's own profiler can bypass the ARM virtual counter.  The
    # remaining inline TSC calibration in full-only native extensions is
    # handled by libimb_clock_shim below.
    kit_args+=("--/plugins/carb.profiler-cpu.plugin/forceMonotonic=true")
fi
if [[ "${show_window}" -eq 1 || -n "${IMB_TEST_CREATE_REFERENCE_URL:-}" ]]; then
    kit_args+=("--ext-folder" "/opt/imb/exts" "--enable" "isaacmetalbridge.input")
fi
if [[ "${demo_scene}" -eq 1 ]]; then
    startup_stage="/opt/imb-scenes/metal-ray-scene.usda"
elif [[ "${simple_grid}" -eq 1 ]]; then
    startup_stage="/opt/imb-grid/default_environment.usd"
fi
if [[ -n "${startup_stage}" ]]; then
    # isaacsim.exp.base.kit creates an empty stage and its legacy
    # /app/content/usdFile opener is disabled in Isaac Sim 6.0.1.  Load the
    # requested stage from a small early extension instead of passing an
    # ignored positional USD argument or replacing the stage after rendering.
    kit_args+=(
        "--/app/content/emptyStageOnStart=false"
        "--/isaac/startup/create_new_stage=false"
        "--ext-folder" "/opt/imb-stage-exts"
        "--enable" "isaacmetalbridge.stage"
    )
fi
kit_args+=("--/crashreporter/uploadToBacktrace=false")
if [[ "${#extra_args[@]}" -gt 0 ]]; then
    kit_args+=("${extra_args[@]}")
fi

echo "run-isaac-sim: starting real Isaac Sim ${experience} experience as ${container_id}"
if [[ "${demo_scene}" -eq 1 ]]; then
    if [[ "${animate_demo}" -eq 1 ]]; then
        echo "run-isaac-sim: opening and playing the animated local Metal validation stage during Kit startup"
    else
        echo "run-isaac-sim: opening local cube/ground Metal validation stage during Kit startup (not NVIDIA Simple Grid)"
    fi
elif [[ "${simple_grid}" -eq 1 ]]; then
    echo "run-isaac-sim: opening cached NVIDIA Simple Grid during Kit startup"
elif [[ "${#extra_args[@]}" -gt 0 ]]; then
    echo "run-isaac-sim: additional Kit arguments supplied verbatim; use --demo-scene or --simple-grid for supported stage selection"
else
    echo "run-isaac-sim: no stage argument supplied; ${experience} opens its default empty stage"
fi

if [[ -n "${IMB_ISAAC_MEMORY:-}" ]]; then
    container_memory="${IMB_ISAAC_MEMORY}"
    memory_source="IMB_ISAAC_MEMORY override"
else
    host_memory_bytes="$(sysctl -n hw.memsize 2>/dev/null || true)"
    if [[ "${host_memory_bytes}" =~ ^[0-9]+$ ]] && (( host_memory_bytes >= 8589934592 )); then
        host_memory_gib=$((host_memory_bytes / 1073741824))
        container_memory_gib=$((host_memory_gib * 3 / 4))
        (( container_memory_gib >= 4 )) || container_memory_gib=4
        container_memory="${container_memory_gib}g"
        memory_source="75% of ${host_memory_gib} GiB host memory"
    else
        container_memory="16g"
        memory_source="portable fallback"
    fi
fi
echo "run-isaac-sim: container memory ${container_memory} (${memory_source}); override with IMB_ISAAC_MEMORY"
if [[ "${sparse_images}" == "1" ]]; then
    echo "run-isaac-sim: Metal sparse-image tile residency enabled"
else
    echo "run-isaac-sim: sparse-image residency disabled by IMB_VULKAN_SPARSE_IMAGES=0"
fi

container_env_args=(
    --env "ACCEPT_EULA=Y"
    --env "RESOURCE_NAME=IsaacSim"
    --env "IMB_VSOCK_PORT=${vsock_port}"
    --env "IMB_VULKAN_NVIDIA_COMPAT=1"
    # Mask 59 matches Isaac Sim 6.0.1's bundled aarch64/DGX Spark shader
    # permutations while leaving synchronization2/robustness2/host-query-reset
    # disabled until their command paths are implemented by the Metal bridge.
    --env "IMB_VULKAN_PROFILE_MASK=${IMB_VULKAN_PROFILE_MASK:-59}"
    # Empty RTX scenes use null acceleration-structure descriptors. This is
    # narrower than enabling the still-incomplete robustness2/synchronization2
    # profile group and is required for updateTopAccelStruct to continue.
    --env "IMB_VULKAN_NULL_DESCRIPTOR=${IMB_VULKAN_NULL_DESCRIPTOR:-1}"
    --env "IMB_VULKAN_SPARSE_IMAGES=${sparse_images}"
)
if [[ "${spirv_compute}" != "0" ]]; then
    container_env_args+=(--env "IMB_VULKAN_SPIRV_COMPUTE=1")
fi
container_env_args+=(--env "IMB_VULKAN_GENERIC_COMPUTE=${generic_compute}")
if [[ -n "${IMB_VULKAN_RT_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_RT_TRACE=${IMB_VULKAN_RT_TRACE}")
fi
container_mount_args=()
if [[ "${demo_scene}" -eq 1 ]]; then
    container_mount_args+=(
        --mount "type=bind,source=${repo_root}/guest/scenes,target=/opt/imb-scenes,readonly"
    )
fi
if [[ "${simple_grid}" -eq 1 ]]; then
    container_mount_args+=(
        --mount "type=bind,source=$(dirname "${simple_grid_cache}"),target=/opt/imb-grid,readonly"
    )
    container_env_args+=(--env "IMB_VULKAN_SCENE_GRID=1")
fi
if [[ -n "${startup_stage}" ]]; then
    container_env_args+=(
        --env "IMB_STARTUP_STAGE_URL=${startup_stage}"
        --env "IMB_CAMERA_STATE_FILE=/opt/imb-camera/state.bin"
        --env "IMB_VULKAN_SCENE_PRESENTATION=1"
    )
    if [[ "${experience}" == "full" && "${show_window}" -eq 1 ]]; then
        container_env_args+=(
            --env "IMB_RESTORE_FULL_LAYOUT=1"
            --env "IMB_VULKAN_FULL_WORKSPACE_ONLY=${IMB_VULKAN_FULL_WORKSPACE_ONLY:-1}"
            --env "IMB_FULL_LAYOUT_READY_FILE=/opt/imb-camera/full-layout-ready"
            --env "IMB_VULKAN_FULL_WORKSPACE_READY_FILE=/opt/imb-camera/full-layout-ready"
        )
    fi
    container_mount_args+=(
        --mount "type=bind,source=${repo_root}/guest/kit_stage_extension/isaacmetalbridge.stage,target=/opt/imb-stage-exts/isaacmetalbridge.stage,readonly"
        --mount "type=bind,source=${camera_dir},target=/opt/imb-camera"
    )
fi
if [[ "${animate_demo}" -eq 1 ]]; then
    container_env_args+=(--env "IMB_TIMELINE_AUTOPLAY=1")
fi
if [[ -n "${camera_sensor_output}" ]]; then
    container_env_args+=(
        --env "IMB_CAMERA_SENSOR_OUTPUT=/opt/imb-camera/sensor-output.ppm"
        --env "IMB_CAMERA_SENSOR_FRAME_FILE=/opt/imb-camera/sensor-frame.ppm"
        --env "IMB_CAMERA_SENSOR_WIDTH=${camera_sensor_width}"
        --env "IMB_CAMERA_SENSOR_HEIGHT=${camera_sensor_height}"
    )
fi
if [[ -n "${physics_smoke_output}" ]]; then
    container_env_args+=(
        --env "IMB_PHYSICS_SMOKE_OUTPUT=/opt/imb-camera/physics-smoke.json"
    )
fi
if [[ "${show_window}" -eq 1 ]]; then
    container_env_args+=(--env "IMB_INPUT_FILE=/opt/imb-runtime/$(basename "${input_output}")")
fi
if [[ "${show_window}" -eq 1 || -n "${IMB_TEST_CREATE_REFERENCE_URL:-}" ]]; then
    container_mount_args+=(
        --mount "type=bind,source=${repo_root}/guest/kit_input_extension/isaacmetalbridge.input,target=/opt/imb/exts/isaacmetalbridge.input,readonly"
    )
fi
if [[ "${show_window}" -eq 1 ]]; then
    container_mount_args+=(
        --mount "type=bind,source=${runtime_dir},target=/opt/imb-runtime,readonly"
    )
    if [[ -n "${IMB_INPUT_TRACE:-}" ]]; then
        container_env_args+=(--env "IMB_INPUT_TRACE=${IMB_INPUT_TRACE}")
    fi
fi
if [[ -n "${IMB_TEST_CREATE_REFERENCE_URL:-}" ]]; then
    container_env_args+=(
        --env "IMB_TEST_CREATE_REFERENCE_URL=${IMB_TEST_CREATE_REFERENCE_URL}"
        --env "IMB_TEST_CREATE_REFERENCE_AFTER_UPDATES=${IMB_TEST_CREATE_REFERENCE_AFTER_UPDATES:-120}"
    )
fi
vk_driver_files="/usr/local/share/vulkan/icd.d/imb_icd.json"
if [[ -n "${IMB_VULKAN_ICD_OVERRIDE:-}" ]]; then
    if [[ ! -d "${IMB_VULKAN_ICD_OVERRIDE}" \
        || ! -f "${IMB_VULKAN_ICD_OVERRIDE}/libimb_vulkan_icd.so" \
        || ! -f "${IMB_VULKAN_ICD_OVERRIDE}/imb_icd.json" ]]; then
        echo "run-isaac-sim: IMB_VULKAN_ICD_OVERRIDE must contain libimb_vulkan_icd.so and imb_icd.json" >&2
        exit 1
    fi
    icd_override="$(cd "${IMB_VULKAN_ICD_OVERRIDE}" && pwd -P)"
    container_mount_args+=(
        --mount "type=bind,source=${icd_override},target=/opt/imb-override,readonly"
    )
    if [[ -f "${IMB_VULKAN_ICD_OVERRIDE}/libcuda.so.1" ]]; then
        container_env_args+=(--env "LD_LIBRARY_PATH=/opt/imb-override:/usr/local/lib")
    fi
    if [[ "${experience}" == "full" \
        && -f "${IMB_VULKAN_ICD_OVERRIDE}/libimb_clock_shim.so" ]]; then
        ld_preload_value="/opt/imb-override/libimb_clock_shim.so"
    fi
    vk_driver_files="/opt/imb-override/imb_icd.json"
fi
container_env_args+=(--env "VK_DRIVER_FILES=${vk_driver_files}")
if [[ -n "${IMB_VULKAN_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_TRACE=${IMB_VULKAN_TRACE}")
fi
if [[ -n "${IMB_VULKAN_UI_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_UI_TRACE=${IMB_VULKAN_UI_TRACE}")
fi
if [[ -n "${IMB_VULKAN_UI_SNAPSHOT_CHANGES:-}" ]]; then
    container_env_args+=(
        --env "IMB_VULKAN_UI_SNAPSHOT_CHANGES=${IMB_VULKAN_UI_SNAPSHOT_CHANGES}"
    )
fi
if [[ -n "${IMB_VULKAN_COMPUTE_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_COMPUTE_TRACE=${IMB_VULKAN_COMPUTE_TRACE}")
fi
if [[ -n "${IMB_VULKAN_RT_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_RT_TRACE=${IMB_VULKAN_RT_TRACE}")
fi
if [[ -n "${IMB_VULKAN_EXTERNAL_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_VULKAN_EXTERNAL_TRACE=${IMB_VULKAN_EXTERNAL_TRACE}")
fi
if [[ -n "${IMB_VULKAN_SEMAPHORE_TRACE:-}" ]]; then
    container_env_args+=(
        --env "IMB_VULKAN_SEMAPHORE_TRACE=${IMB_VULKAN_SEMAPHORE_TRACE}"
    )
fi
if [[ -n "${IMB_CUDA_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_CUDA_TRACE=${IMB_CUDA_TRACE}")
fi
if [[ -n "${IMB_VULKAN_SHADER_DUMP_DIR:-}" ]]; then
    mkdir -p "${IMB_VULKAN_SHADER_DUMP_DIR}"
    shader_dump_host_dir="$(cd "${IMB_VULKAN_SHADER_DUMP_DIR}" && pwd -P)"
    container_mount_args+=(
        --mount "type=bind,source=${shader_dump_host_dir},target=/opt/imb-shader-dump"
    )
    container_env_args+=(--env "IMB_VULKAN_SHADER_DUMP_DIR=/opt/imb-shader-dump")
fi
if [[ -n "${IMB_FILE_OPEN_TRACE_LIBRARY:-}" ]]; then
    if [[ ! -f "${IMB_FILE_OPEN_TRACE_LIBRARY}" ]]; then
        echo "run-isaac-sim: IMB_FILE_OPEN_TRACE_LIBRARY must be a shared library" >&2
        exit 1
    fi
    file_open_trace_directory="$(cd "$(dirname "${IMB_FILE_OPEN_TRACE_LIBRARY}")" && pwd -P)"
    file_open_trace_name="$(basename "${IMB_FILE_OPEN_TRACE_LIBRARY}")"
    container_mount_args+=(
        --mount "type=bind,source=${file_open_trace_directory},target=/opt/imb-file-open-trace,readonly"
    )
    if [[ -n "${ld_preload_value}" ]]; then
        ld_preload_value="${ld_preload_value}:/opt/imb-file-open-trace/${file_open_trace_name}"
    else
        ld_preload_value="/opt/imb-file-open-trace/${file_open_trace_name}"
    fi
fi
if [[ -n "${ld_preload_value}" ]]; then
    container_env_args+=(--env "LD_PRELOAD=${ld_preload_value}")
fi
if [[ -n "${IMB_CLOCK_SHIM_TRACE:-}" ]]; then
    container_env_args+=(--env "IMB_CLOCK_SHIM_TRACE=${IMB_CLOCK_SHIM_TRACE}")
fi
if [[ -n "${IMB_SHADER_CACHE_OVERRIDE:-}" ]]; then
    if [[ ! -d "${IMB_SHADER_CACHE_OVERRIDE}" ]]; then
        echo "run-isaac-sim: IMB_SHADER_CACHE_OVERRIDE must be a directory" >&2
        exit 1
    fi
    shader_cache_override="$(cd "${IMB_SHADER_CACHE_OVERRIDE}" && pwd -P)"
    # Isaac Sim 6.0.1 constructs ShaderDb's cache search list from the bundled
    # extension path before command-line settings are consumed.  Stage the
    # small override into the container's writable root layer before Kit starts.
    # ShaderDb performs low-level reads that do not work reliably from a direct
    # VirtioFS overlay, while files copied into the image filesystem do.
    shader_cache_container_dir="/isaac-sim/extscache/omni.gpu_foundation.shadercache.vulkan-1.0.0+f9bf0dda.la64.r/cache/shadercache"
    container_mount_args+=(
        --mount "type=bind,source=${shader_cache_override},target=/opt/imb-shader-cache,readonly"
    )
    shader_cache_stage_command="cp -a /opt/imb-shader-cache/. ${shader_cache_container_dir}/ && exec \"\$@\""
    kit_entrypoint="${entrypoint}"
    entrypoint="/bin/sh"
    kit_args=(-c "${shader_cache_stage_command}" imb-shader-cache-stage "${kit_entrypoint}" "${kit_args[@]}")
fi
container_run_args=(
    --detach
    --name "${container_id}"
    --platform linux/arm64
    --cpus "${IMB_ISAAC_CPUS:-8}"
    --memory "${container_memory}"
    --shm-size "${IMB_ISAAC_SHM_SIZE:-1g}"
    "${container_env_args[@]}"
)
if [[ "${#container_mount_args[@]}" -gt 0 ]]; then
    container_run_args+=("${container_mount_args[@]}")
fi
container_run_args+=(
    --entrypoint "${entrypoint}"
    "${derived_image}"
    "${kit_args[@]}"
)
container run "${container_run_args[@]}" >/dev/null

listener_ready=0
for _ in {1..240}; do
    if container logs "${container_id}" 2>&1 | grep -F "listening on vsock port ${vsock_port}" >/dev/null; then
        listener_ready=1
        break
    fi
    if ! container inspect "${container_id}" 2>/dev/null | grep -F '"state" : "running"' >/dev/null; then
        break
    fi
    sleep 0.25
done

if [[ "${listener_ready}" -ne 1 ]]; then
    echo "run-isaac-sim: guest bridge listener did not become ready" >&2
    container logs "${container_id}" 2>&1 | tail -n 160 >&2 || true
    exit 1
fi

container logs --follow -n 0 "${container_id}" &
log_pid=$!

# The interactive viewer intentionally runs until Ctrl-C. Publish a completed
# one-shot sensor artifact to the requested host path as soon as it exists,
# rather than making the user stop Isaac before the two files are copied.
if [[ -n "${camera_sensor_output}" ]]; then
    (
        temporary_output="${camera_sensor_output}.tmp.${container_id}"
        temporary_metadata="${camera_sensor_output}.json.tmp.${container_id}"
        while true; do
            if [[ -s "${camera_sensor_staged_output}" \
                && -s "${camera_sensor_staged_output}.json" ]]; then
                cp "${camera_sensor_staged_output}" "${temporary_output}"
                cp "${camera_sensor_staged_output}.json" "${temporary_metadata}"
                mv "${temporary_output}" "${camera_sensor_output}"
                mv "${temporary_metadata}" "${camera_sensor_output}.json"
                exit 0
            fi
            sleep 0.2
        done
    ) &
    sensor_copy_pid=$!
fi
if [[ -n "${physics_smoke_output}" ]]; then
    (
        temporary_output="${physics_smoke_output}.tmp.${container_id}"
        while true; do
            if [[ -s "${physics_smoke_staged_output}" ]]; then
                cp "${physics_smoke_staged_output}" "${temporary_output}"
                mv "${temporary_output}" "${physics_smoke_output}"
                exit 0
            fi
            sleep 0.2
        done
    ) &
    physics_copy_pid=$!
fi

adapter_status=0
if [[ -n "${frame_output}" ]]; then
    IMB_SCENE_MATERIAL_TEXTURE="${scene_material_texture}" \
    IMB_FRAME_OUTPUT="${frame_output}" \
    IMB_FRAME_OUTPUT_UI_ONLY="${show_window}" \
    IMB_FRAME_OUTPUT_ALL="${IMB_FRAME_OUTPUT_ALL:-0}" \
    IMB_FRAME_OUTPUT_ALL_LATEST="${IMB_FRAME_OUTPUT_ALL_LATEST:-0}" \
    "${adapter}" \
        --container "${container_id}" \
        --vsock-port "${vsock_port}" \
        --sessions "${adapter_sessions}" \
        --wait-exit || adapter_status=$?
else
    IMB_SCENE_MATERIAL_TEXTURE="${scene_material_texture}" \
    "${adapter}" \
        --container "${container_id}" \
        --vsock-port "${vsock_port}" \
        --sessions "${adapter_sessions}" \
        --wait-exit || adapter_status=$?
fi

kill "${log_pid}" >/dev/null 2>&1 || true
wait "${log_pid}" >/dev/null 2>&1 || true
log_pid=""
if [[ -n "${sensor_copy_pid}" ]]; then
    kill "${sensor_copy_pid}" >/dev/null 2>&1 || true
    wait "${sensor_copy_pid}" >/dev/null 2>&1 || true
    sensor_copy_pid=""
fi
if [[ -n "${physics_copy_pid}" ]]; then
    kill "${physics_copy_pid}" >/dev/null 2>&1 || true
    wait "${physics_copy_pid}" >/dev/null 2>&1 || true
    physics_copy_pid=""
fi

if [[ "${adapter_status}" -ne 0 ]]; then
    if container logs "${container_id}" 2>&1 | grep -F 'app ready' >/dev/null; then
        echo "run-isaac-sim: Isaac Sim crashed or exited with status ${adapter_status} after reporting app ready" >&2
    else
        echo "run-isaac-sim: Isaac Sim/container exited with status ${adapter_status} before app ready" >&2
    fi
    container logs "${container_id}" 2>&1 | tail -n 200 >&2 || true
    exit "${adapter_status}"
fi
if ! container logs "${container_id}" 2>&1 | grep -F 'app ready' >/dev/null; then
    echo "run-isaac-sim: Isaac Sim exited before reporting app ready" >&2
    container logs "${container_id}" 2>&1 | tail -n 200 >&2 || true
    exit 1
fi

echo "run-isaac-sim: real Isaac Sim reported app ready through the Apple/Metal bridge"
if [[ -n "${camera_sensor_output}" ]]; then
    if [[ ! -s "${camera_sensor_staged_output}" \
        || ! -s "${camera_sensor_staged_output}.json" ]]; then
        echo "run-isaac-sim: Isaac Camera sensor did not publish its requested output" >&2
        exit 1
    fi
    cp "${camera_sensor_staged_output}" "${camera_sensor_output}"
    cp "${camera_sensor_staged_output}.json" "${camera_sensor_output}.json"
    echo "run-isaac-sim: Isaac Camera sensor output ${camera_sensor_output}"
fi
if [[ -n "${physics_smoke_output}" ]]; then
    if [[ ! -s "${physics_smoke_staged_output}" ]]; then
        echo "run-isaac-sim: CPU PhysX smoke test did not publish its requested output" >&2
        exit 1
    fi
    cp "${physics_smoke_staged_output}" "${physics_smoke_output}"
    if ! grep -F '"passed": true' "${physics_smoke_output}" >/dev/null; then
        echo "run-isaac-sim: CPU PhysX smoke test reported failure" >&2
        cat "${physics_smoke_output}" >&2
        exit 1
    fi
    echo "run-isaac-sim: CPU PhysX smoke output ${physics_smoke_output}"
fi
if [[ "${show_window}" -eq 1 ]]; then
    echo "run-isaac-sim: real Isaac Sim UI was displayed with native pointer and keyboard forwarding"
fi
