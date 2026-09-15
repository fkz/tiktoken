#!/usr/bin/env bash
# Synthetic physical-core scaling; does not change memory-channel configuration.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "$root/perf-results"
out=$(mktemp -d "$root/perf-results/channel-check-XXXXXXXX")
echo "Results: $out"
export ZIG_GLOBAL_CACHE_DIR=${ZIG_GLOBAL_CACHE_DIR:-"$root/.zig-cache/perf-global"}
"${ZIG:-zig}" cc -O3 -mavx2 -pthread "$root/tools/perf/stream-read.c" -o "$out/stream-read"
exec "${PYTHON:-python3}" "$root/tools/perf/channel-check.py" "$out" "$@"
