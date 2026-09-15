#!/usr/bin/env bash
prefetch_source=all
conversion=--amd-prefetch
cache_level=1
cache_type=Data
if [[ ${1-} == --dram ]]; then
    prefetch_source=dram_io_all
    conversion=--amd-prefetch-dram
    shift
elif [[ ${1-} == --l2 ]]; then
    conversion=--amd-prefetch-l2
    cache_level=2
    cache_type=Unified
    shift
fi
source "$(dirname -- "$0")/common.sh"
require_perf
command -v "${PYTHON:-python3}" >/dev/null || { echo 'Python 3 is required.' >&2; exit 2; }
export LC_ALL=C

# Keep the group small enough to run without multiplexing.
event_names=()
for family in ls_dmnd_fills_from_sys ls_hw_pf_dc_fills ls_sw_pf_dc_fills; do
    event_names+=("$family.$prefetch_source")
done
if [[ $cache_level == 2 ]]; then
    event_names=(l2_fill_rsp_src.dram_io_near l2_fill_rsp_src.dram_io_far
        l2_pf_miss_l2_l3.l1_dc_l2_hwpf ls_sw_pf_dc_fills.dram_io_all)
fi
events=()
for event in "${event_names[@]}"; do
    "$perf_bin" list --details "$event" > "$out/event-check.txt"
    if ! grep -Fq "$event" "$out/event-check.txt"; then
        echo "Unsupported prefetch counter: $event" | tee "$out/unavailable.txt"
        exit 2
    fi
    cat "$out/event-check.txt" >> "$out/events.txt"
    events+=("$event:u")
done
rm "$out/event-check.txt"

# Verify the line size used to convert fill counts into bytes.
found_cache=false
for cache in /sys/devices/system/cpu/cpu[0-9]*/cache/index*; do
    if [[ $(cat "$cache/level") == "$cache_level" && $(cat "$cache/type") == "$cache_type" ]]; then
        found_cache=true
        [[ $(cat "$cache/coherency_line_size") == 64 ]] || {
            echo "Unsupported L$cache_level cache line size." >&2
            exit 2
        }
    fi
done
[[ $found_cache == true ]] || { echo "Cannot determine L$cache_level cache line size." >&2; exit 2; }
echo "L$cache_level cache line size: 64 bytes" >> "$out/events.txt"
group=$(IFS=,; echo "${events[*]}")
"$perf_bin" stat -I 100 -x ';' -o "$out/prefetch.txt" -e "{$group}" \
    -- "${command_args[@]}" > "$out/stdout.txt" 2> "$out/stderr.txt" || {
        cat "$out/prefetch.txt" "$out/stderr.txt" >&2
        exit 1
    }
"${PYTHON:-python3}" "$root/tools/perf/bandwidth.py" "$out/prefetch.txt" "$conversion" | tee "$out/rates.txt"
