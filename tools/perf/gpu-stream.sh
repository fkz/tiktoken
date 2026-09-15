#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$root/perf-results"
out=$(mktemp -d "$root/perf-results/gpu-stream-XXXXXXXX")
echo "Results: $out"
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-"$root/.zig-cache/perf-global"}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-"$root/.zig-cache/mesa"}
if [[ -n ${VULKAN_INCLUDE:-} && -n ${VULKAN_LIB:-} ]]; then
    vk_flags=(-I "$VULKAN_INCLUDE" -L "$VULKAN_LIB" -Wl,-rpath,"$VULKAN_LIB" -lvulkan)
else
    command -v pkg-config >/dev/null || { echo 'Install Vulkan development files and pkg-config, or set VULKAN_INCLUDE and VULKAN_LIB.' >&2; exit 2; }
    read -r -a vk_flags <<< "$(pkg-config --cflags --libs vulkan)"
    (( ${#vk_flags[@]} )) || { echo 'Vulkan development files are required.' >&2; exit 2; }
fi
"${GLSLANG:-glslangValidator}" -V --target-env vulkan1.1 "$root/tools/perf/gpu-stream.comp" -o "$out/read.spv"
"${ZIG:-zig}" cc -O3 -Wall -Wextra -pthread "$root/tools/perf/gpu-stream.c" "${vk_flags[@]}" -o "$out/gpu-stream"
"${ZIG:-zig}" cc -O3 -Wall -Wextra -mavx2 -pthread "$root/tools/perf/stream-read.c" -o "$out/cpu-stream"
sha256sum "$out/read.spv" "$out/gpu-stream" "$out/cpu-stream" > "$out/binaries.sha256"
exec "${PYTHON:-python3}" "$root/tools/perf/gpu-stream.py" "$out" "$@"
