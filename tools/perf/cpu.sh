#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
require_perf
# Keep groups small; perf reports counter multiplexing in the saved output.
"$perf_bin" stat -o "$out/cpu.txt" \
    -e task-clock,context-switches,cpu-migrations,page-faults \
    -e '{cycles:u,instructions:u}' -e cache-references:u,cache-misses:u \
    -- "${command_args[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt" || {
        cat "$out/cpu.txt" "$out/stderr.txt" >&2
        exit 1
    }
cat "$out/cpu.txt"
