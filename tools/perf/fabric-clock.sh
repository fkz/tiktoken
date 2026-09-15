#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
exec "${PYTHON:-python3}" "$root/tools/perf/fabric-clock.py" "$out" "${command_args[@]}"
