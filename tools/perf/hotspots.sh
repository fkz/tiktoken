#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
require_perf
"$perf_bin" record -o "$out/perf.data" -e cycles:u -F 199 --call-graph dwarf,8192 \
    -- "${command_args[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt" || {
        cat "$out/stderr.txt" >&2
        exit 1
    }
"$perf_bin" report --stdio --no-children -i "$out/perf.data" > "$out/hotspots.txt"
head -60 "$out/hotspots.txt"
