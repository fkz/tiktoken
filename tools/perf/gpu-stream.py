#!/usr/bin/env python3
"""Measure CPU alone, integrated GPU alone, and both over coordinated intervals."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import select
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--seconds", type=float, default=3)
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()
if not 1 <= args.seconds <= 30 or not 1 <= args.repeats <= 10:
    parser.error("Use 1–30 seconds and 1–10 repeats")
cores = {}
for cpu in sorted(os.sched_getaffinity(0)):
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    key = tuple((base / "topology" / name).read_text().strip() for name in ("physical_package_id", "core_id"))
    freq = base / "cpufreq/cpuinfo_max_freq"
    cores.setdefault(key, (int(freq.read_text()) if freq.exists() else 0, cpu))
cpus = [cpu for freq, cpu in sorted(cores.values(), key=lambda p: (-p[0], p[1]))[:2]]
commands = {"cpu": [str(args.output / "cpu-stream"), *map(str, cpus)],
            "gpu": [str(args.output / "gpu-stream"), str(args.output / "read.spv")]}
env = {**os.environ, "STREAM_SECONDS": str(args.seconds), "STREAM_SYNC": "1"}
(args.output / "environment.json").write_text(json.dumps({"platform": platform.platform(), "commands": commands,
    "seconds": args.seconds, "repeats": args.repeats, "cpu_ids": cpus,
    "VK_DRIVER_FILES": env.get("VK_DRIVER_FILES")}, indent=2) + "\n")
runs = []
order = ["cpu", "gpu", "both"] * args.repeats
random.Random(42).shuffle(order)
for index, mode in enumerate(order):
    active = ["cpu", "gpu"] if mode == "both" else [mode]
    children, logs = {}, []
    try:
        for name in active:
            log = (args.output / f"{index}-{mode}-{name}.stderr").open("w")
            logs.append(log)
            children[name] = subprocess.Popen(commands[name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                              stderr=log, text=True, env=env)
        for name, child in children.items():
            if not select.select([child.stdout], [], [], 30)[0] or child.stdout.readline().strip() != "READY":
                raise RuntimeError(f"{name} failed to initialize; see {index}-{mode}-{name}.stderr")
        target = time.monotonic() + 0.2
        for child in children.values():
            child.stdin.write(f"{target:.9f}\n")
            child.stdin.close()
            child.stdin = None
        results = {}
        for name, child in children.items():
            stdout, _ = child.communicate(timeout=args.seconds + 30)
            if child.returncode:
                raise RuntimeError(f"{name} failed; see {index}-{mode}-{name}.stderr")
            results[name] = json.loads(stdout)
        first = min(r["start_s"] for r in results.values())
        last = max(r["end_s"] for r in results.values())
        overlap = min(r["end_s"] for r in results.values()) - max(r["start_s"] for r in results.values())
        overlap_fraction = overlap / (last - first)
        if overlap_fraction < 0.95:
            raise RuntimeError("Measured CPU/GPU intervals overlap by less than 95%; refusing to add rates")
        record = {"mode": mode, "results": results, "common_window_seconds": last - first,
                  "overlap_fraction": overlap_fraction,
                  "combined_GB_s": sum(r["read_bytes"] for r in results.values()) / (last - first) / 1e9}
        runs.append(record)
        (args.output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
        components = ", ".join(f"{n}: {r['GB_s']:.2f}" for n, r in results.items())
        print(f"{mode}: {record['combined_GB_s']:.2f} GB/s ({components}; overlap {overlap_fraction:.1%})", flush=True)
    finally:
        for child in children.values():
            if child.poll() is None:
                child.kill()
            child.wait()
        for log in logs:
            log.close()

summary = {}
for mode in ("cpu", "gpu", "both"):
    selected = [r for r in runs if r["mode"] == mode]
    rates = [r["combined_GB_s"] for r in selected]
    summary[mode] = {"median_GB_s": statistics.median(rates), "min_GB_s": min(rates), "max_GB_s": max(rates)}
    for name in ("cpu", "gpu"):
        components = [r["results"][name]["GB_s"] for r in selected if name in r["results"]]
        if components:
            summary[mode][name + "_median_GB_s"] = statistics.median(components)
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\nMedian sustained read GB/s:")
for mode, result in summary.items():
    print(f"{mode:4}: {result['median_GB_s']:.2f} (range {result['min_GB_s']:.2f}–{result['max_GB_s']:.2f})")
print("Aggregate uses completed read bytes / common wall-clock window, not summed GPU timestamp rates.")
