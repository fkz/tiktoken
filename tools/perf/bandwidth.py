#!/usr/bin/env python3
"""Convert perf interval traffic, retaining its explicit units, to GiB/s."""
import pathlib
import sys
import math

amd_l2 = sys.argv[2:] == ["--amd-l2"]
amd_prefetch = sys.argv[2:] in (["--amd-prefetch"], ["--amd-prefetch-dram"])
amd_prefetch_l2 = sys.argv[2:] == ["--amd-prefetch-l2"]
prefetch_source = "dram_io_all" if sys.argv[2:] == ["--amd-prefetch-dram"] else "all"
if len(sys.argv) < 2 or (sys.argv[2:] and not (amd_l2 or amd_prefetch or amd_prefetch_l2)):
    sys.exit("Usage: bandwidth.py PERF_REPORT [--amd-l2|--amd-prefetch|--amd-prefetch-dram|--amd-prefetch-l2]")
prefetch_events = [
    f"{family}.{prefetch_source}:u"
    for family in ("ls_dmnd_fills_from_sys", "ls_hw_pf_dc_fills", "ls_sw_pf_dc_fills")
]
prefetch_l2_events = [
    "l2_fill_rsp_src.dram_io_near:u", "l2_fill_rsp_src.dram_io_far:u",
    "l2_pf_miss_l2_l3.l1_dc_l2_hwpf:u", "ls_sw_pf_dc_fills.dram_io_all:u",
]
intervals = {}
event_sets = {}
scales = {"bytes": 1, "Bytes": 1, "MiB": 2**20, "GiB": 2**30,
          "MB": 10**6, "GB": 10**9}
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = [field.strip() for field in line.split(";")]
    if len(row) < 4:
        continue
    stamp, count, unit, event = row[:4]
    if amd_prefetch_l2:
        if event not in prefetch_l2_events:
            continue
        if unit:
            sys.exit(f"Expected unscaled counts, got unit {unit!r}.")
        column = prefetch_l2_events.index(event)
        # Only the L2 fill counters become bytes. Keep requests and L1 fills as counts.
        scale = 64 if column < 2 else 1
    elif amd_prefetch:
        if event not in prefetch_events:
            continue
        if unit:
            sys.exit(f"Expected unscaled fill counts, got unit {unit!r}.")
        scale = 64
        column = prefetch_events.index(event)
    elif amd_l2:
        if event not in ("l2_fill_rsp_src.dram_io_near:u", "l2_fill_rsp_src.dram_io_far:u"):
            continue
        if unit:
            sys.exit(f"Expected unscaled fill counts, got unit {unit!r}.")
        scale = 64
        column = 0 if "near" in event else 1
    else:
        if "cas_count_" not in event:
            continue
        if unit not in scales:
            sys.exit(f"Cannot convert event {event}: unknown unit {unit!r}; inspect raw report.")
        scale = scales[unit]
        column = 0 if "cas_count_read" in event else 1
    try:
        stamp, count = float(stamp), float(count)
    except ValueError:
        sys.exit(f"Counter unavailable: {line}")
    if not math.isfinite(stamp) or not math.isfinite(count) or count < 0:
        sys.exit(f"Invalid counter value: {line}")
    if len(row) < 6:
        sys.exit(f"Missing counter coverage: {line}")
    try:
        coverage = float(row[5].rstrip("%"))
    except ValueError:
        sys.exit(f"Invalid counter coverage: {line}")
    if not math.isfinite(coverage) or coverage < 99:
        sys.exit(f"Counter ran for less than 99% of enabled time; retry without competing profilers: {line}")
    events = event_sets.setdefault(stamp, set())
    if event in events:
        sys.exit(f"Duplicate event in interval: {line}")
    events.add(event)
    traffic = intervals.setdefault(stamp, [0.0] * (4 if amd_prefetch_l2 else 3 if amd_prefetch else 2))
    traffic[column] += count * scale

if not intervals:
    sys.exit("No measured traffic found; inspect raw report.")
expected = (set(prefetch_l2_events) if amd_prefetch_l2 else
            set(prefetch_events) if amd_prefetch else
            set(("l2_fill_rsp_src.dram_io_near:u", "l2_fill_rsp_src.dram_io_far:u"))
            if amd_l2 else set.union(*event_sets.values()))
if any(events != expected for events in event_sets.values()):
    sys.exit("Incomplete set of counters in an interval; inspect raw report.")
if amd_prefetch_l2:
    print("AMD DRAM/MMIO-to-L2 fills and prefetch activity, command and threads, user space.")
    print("DRAM_L2: completed L2 fills, including prefetched lines that never reach L1.")
    print("HW_PF: L1/L2 hardware-prefetch requests missing both L2 and L3; not completed DRAM fills.")
    print("SW_PF_L1: software-prefetch L1 fills whose reported source is DRAM/MMIO.")
    print("These counters are not disjoint: do not add them or subtract prefetch counts from total fills.")
    print("elapsed_s  DRAM_L2_GiB/s  HW_PF_Mreq/s  SW_PF_L1_Mfill/s")
    totals = [0.0] * 4
    previous = 0.0
    for stamp, traffic in sorted(intervals.items()):
        duration = stamp - previous
        if duration <= 0:
            sys.exit("Invalid interval duration in perf output.")
        near, far, hw_requests, sw_fills = traffic
        print(f"{stamp:9.3f} {(near + far) / duration / 2**30:14.3f}"
              f" {hw_requests / duration / 1e6:13.3f} {sw_fills / duration / 1e6:16.3f}")
        totals = [a + b for a, b in zip(totals, traffic)]
        previous = stamp
    reads = totals[0] + totals[1]
    print(f"\nAverage over {previous:.3f} s (includes setup and teardown):")
    print(f"DRAM-to-L2: {reads / 64 / 1e6:.3f} million fills; {reads / previous / 2**30:.3f} GiB/s;"
          f" {reads / previous / 1e9:.3f} GB/s")
    print(f"Hardware PF missing L2/L3: {totals[2] / 1e6:.3f} million requests;"
          f" {totals[2] / previous / 1e6:.3f} Mreq/s")
    print(f"Software PF DRAM-to-L1: {totals[3] / 1e6:.3f} million fills;"
          f" {totals[3] / previous / 1e6:.3f} Mfill/s")
    print("Exact prefetched/non-prefetched DRAM fill shares are unavailable from these events.")
    print("HW misses do not identify the final data source; SW fills omit prefetches stopping in L2/L3.")
    print("L3-only fills, writebacks, kernel and GPU traffic are outside the DRAM-to-L2 estimate.")
    sys.exit(0)
if amd_prefetch:
    source_label = "All sources (including L2/L3 caches)" if prefetch_source == "all" else "DRAM/MMIO sources only"
    print("AMD L1 data-cache fills by request type, command and threads, user space.")
    print(source_label)
    print("Rates are GiB/s equivalents: fills x 64 bytes / wall time; includes setup and teardown.")
    print("Demand = demand-triggered L1 fill; HW_PF = hardware prefetch; SW_PF = software prefetch.")
    print("Demand does not mean never prefetched: the line may have been prefetched into L2/L3.")
    print("This is L1 fill activity, not total DRAM-to-L2 bandwidth or all memory accesses.")
    print("elapsed_s     demand      HW_PF      SW_PF (GiB/s equivalents)")
    previous = 0.0
    totals = [0.0] * 3
    for stamp, traffic in sorted(intervals.items()):
        duration = stamp - previous
        if duration <= 0:
            sys.exit("Invalid interval duration in perf output.")
        rates = [value / duration / 2**30 for value in traffic]
        print(f"{stamp:9.3f} " + " ".join(f"{r:10.3f}" for r in rates))
        totals = [a + b for a, b in zip(totals, traffic)]
        previous = stamp
    print(f"\nAverage over {previous:.3f} s:")
    print("Type             million fills   equiv. GiB/s   share of fills")
    total = sum(totals)
    for kind, value in zip(("Demand", "Hardware PF", "Software PF"), totals):
        share = f"{value / total:8.2%}" if total else "     n/a"
        print(f"{kind:16} {value / 64 / 1e6:13.3f}"
              f" {value / previous / 2**30:14.3f} {share:>16}")
    print("Shares classify observed L1 fills; they are not prefetch accuracy or coverage of all loads.")
    sys.exit(0)
if amd_l2:
    print("elapsed_s  near_GiB/s  far_GiB/s    read_GiB/s (DRAM-to-L2 estimate)")
else:
    print("elapsed_s  read_GiB/s  write_GiB/s  total_GiB/s (system-wide)")
previous = 0.0
total = 0.0
for stamp, (reads, writes) in sorted(intervals.items()):
    duration = stamp - previous
    if duration <= 0:
        sys.exit("Invalid interval duration in perf output.")
    factor = duration * 2**30
    print(f"{stamp:9.3f}  {reads/factor:10.3f}  {writes/factor:11.3f}  {(reads+writes)/factor:11.3f}")
    total += reads + writes
    previous = stamp
print(f"Average over {previous:.3f} s: {total / previous / 2**30:.3f} GiB/s")
print(f"Average: {total / previous / 10**9:.3f} GB/s; transferred: {total / 2**30:.3f} GiB")
