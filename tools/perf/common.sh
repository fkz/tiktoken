#!/usr/bin/env bash
# Sourced by the profiling entry points. Build separately to exclude compilation.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 MODEL.gguf [PROMPT (default: Hello)]" >&2
    exit 2
fi
model=$(realpath -- "$1")
[[ -f "$model" ]] || { echo "Model not found: $model" >&2; exit 2; }
binary=${BINARY:-"$root/zig-out/bin/tiktoken"}
binary=$(realpath -- "$binary")
[[ -x "$binary" ]] || { echo 'Run zig build -Doptimize=ReleaseFast first.' >&2; exit 2; }
cd "$root"
command_args=("$binary" continue "$model" "${2-Hello}")
mkdir -p "$root/perf-results"
out=$(mktemp -d "$root/perf-results/$(basename "$0" .sh)-XXXXXXXX")
printf '%q ' "${command_args[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
{ date -Is; uname -a; git rev-parse HEAD; sha256sum "$binary"; } > "$out/environment.txt"
echo "Results: $out"

require_perf() {
    perf_bin=${PERF:-perf}
    command -v "$perf_bin" >/dev/null || {
        echo 'perf is required; on NixOS: nix shell nixpkgs#linuxPackages.perf' >&2
        exit 2
    }
}
