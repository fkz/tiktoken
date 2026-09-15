#!/usr/bin/env python3
"""Compare streaming bandwidth against a user-specified 64-bit channel ceiling."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--mt-s", type=int, required=True, help="Configured memory speed in MT/s")
args = parser.parse_args()
if args.mt_s <= 0:
    parser.error("--mt-s must be positive")

# Choose one logical CPU per physical core, fastest cores first on hybrid Ryzen.
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    key = tuple((base / "topology" / name).read_text().strip()
                for name in ("physical_package_id", "core_id"))
    freq = base / "cpufreq/cpuinfo_max_freq"
    cores.setdefault(key, (int(freq.read_text()) if freq.exists() else 0, cpu))
cpus = [cpu for freq, cpu in sorted(cores.values(), key=lambda pair: (-pair[0], pair[1]))]
counts = sorted(set(n for n in (1, 2, 4, len(cpus)) if n <= len(cpus) and n <= 64))
single = args.mt_s * 8 / 1000
metadata = {"platform": platform.platform(), "cpus": cpus,
            "configured_MT_s": args.mt_s, "single_channel_GB_s": single,
            "method": "1 GiB total, 32 read passes, disjoint ranges, pinned physical cores, 3 repeats"}
(args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Physical core CPU IDs (fastest first): {cpus}", flush=True)
print(f"Theoretical 64-bit channel: {single:.1f} GB/s; two: {2*single:.1f} GB/s", flush=True)
print("These are worker counts, not memory-channel counts.", flush=True)
binary = str(args.output / "stream-read")
# Untimed warm-up, followed by deterministic shuffled run order to reduce drift.
subprocess.run([binary, str(cpus[0])], capture_output=True, text=True, check=True)
order = counts * 3
random.Random(42).shuffle(order)
runs = []
for n in order:
    command = [binary, *map(str, cpus[:n])]
    result = json.loads(subprocess.check_output(command, text=True))
    result["cpus"] = cpus[:n]
    runs.append(result)
    (args.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    print(f"{n} workers: {result['GB_s']:.2f} GB/s", flush=True)

summary = []
baseline = statistics.median(r["GB_s"] for r in runs if r["workers"] == 1)
print("\nworkers  median GB/s  min–max GB/s    vs 1 worker")
for n in counts:
    rates = [r["GB_s"] for r in runs if r["workers"] == n]
    median = statistics.median(rates)
    summary.append({"workers": n, "median_GB_s": median, "min_GB_s": min(rates),
                    "max_GB_s": max(rates), "speedup_vs_one_worker": median / baseline})
    print(f"{n:7d}  {median:11.2f}  {min(rates):5.2f}–{max(rates):5.2f}    {median/baseline:.2f}x")
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
best = max(s["median_GB_s"] for s in summary)
print(f"\nBest median / single-channel theoretical maximum: {best/single:.2f}x")
print("Above 1.0x is evidence of bandwidth beyond one 64-bit channel; below is inconclusive.")
print("Exact dual-versus-single-channel speedup requires rerunning with one channel disabled.")
