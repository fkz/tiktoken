#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
exec "${PYTHON:-python3}" "$root/tools/perf/memory.py" "$out" "${command_args[@]}"
