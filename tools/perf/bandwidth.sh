#!/usr/bin/env bash
if [[ ${1-} == --prefetch ]]; then
    shift
    exec "$(dirname -- "$0")/prefetch.sh" "$@"
fi
source "$(dirname -- "$0")/common.sh"
require_perf
command -v "${PYTHON:-python3}" >/dev/null || { echo 'Python 3 is required.' >&2; exit 2; }
export LC_ALL=C
# Prefer controller traffic; otherwise use AMD's source-qualified L2 fills.
events=()
scope=(-a)
conversion=()
shopt -s nullglob
for pmu in /sys/bus/event_source/devices/uncore_imc*; do
    if [[ -f "$pmu/events/cas_count_read" && -f "$pmu/events/cas_count_write" ]]; then
        events+=(-e "$(basename "$pmu")/cas_count_read/" -e "$(basename "$pmu")/cas_count_write/")
        for event in "$pmu"/events/cas_count_{read,write}*; do
            printf '%s: %s\n' "$event" "$(cat "$event")" >> "$out/events.txt"
        done
    fi
done
if (( ${#events[@]} == 0 )); then
    "$perf_bin" list --details 'l2_fill_rsp_src.*' > "$out/events.txt"
    if ! grep -q 'l2_fill_rsp_src.dram_io_near' "$out/events.txt" ||
       ! grep -q 'l2_fill_rsp_src.dram_io_far' "$out/events.txt"; then
        echo 'No supported IMC or AMD L2 DRAM-source events found.' | tee "$out/unavailable.txt"
        exit 2
    fi
    # Check the line size on every online L2 cache rather than assuming a scale.
    line_sizes=()
    for cache in /sys/devices/system/cpu/cpu[0-9]*/cache/index*; do
        if [[ $(cat "$cache/level") == 2 ]]; then
            line_sizes+=("$(cat "$cache/coherency_line_size")")
        fi
    done
    if (( ${#line_sizes[@]} == 0 )); then
        echo 'Cannot determine L2 cache line size.' >&2
        exit 2
    fi
    for size in "${line_sizes[@]}"; do
        [[ $size == 64 ]] || { echo "Unsupported L2 line size: $size" >&2; exit 2; }
    done
    echo 'L2 cache line size: 64 bytes' >> "$out/events.txt"
    events=(-e '{l2_fill_rsp_src.dram_io_near:u,l2_fill_rsp_src.dram_io_far:u}')
    scope=()
    conversion=(--amd-l2)
    echo 'AMD DRAM-to-L2 read bandwidth estimate, counted for the command and its threads.'
    echo 'Includes DRAM/MMIO-sourced fills; excludes writebacks, kernel and GPU traffic.'
else
    echo 'Measuring system-wide DRAM traffic, including other processes. Keep the machine idle.'
fi
"$perf_bin" stat "${scope[@]}" -I 100 -x ';' -o "$out/bandwidth.txt" "${events[@]}" \
    -- "${command_args[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt" || {
        cat "$out/bandwidth.txt" "$out/stderr.txt" >&2
        exit 1
    }
"${PYTHON:-python3}" "$root/tools/perf/bandwidth.py" "$out/bandwidth.txt" "${conversion[@]}" | tee "$out/rates.txt"
