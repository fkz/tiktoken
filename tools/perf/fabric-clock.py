#!/usr/bin/env python3
"""Read AMD APU fabric telemetry before, during three command runs, and after."""
import csv
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

# Linux drivers/gpu/drm/amd/include/kgd_pp_interface.h, gpu_metrics_v3_0.
# Native x86-64 ABI offsets; accept only the exact 264-byte version 3.0 table.
# average_fclk_frequency is firmware time-filtered MHz, not an instant clock.
def read_metrics(path):
    data = path.read_bytes()
    if len(data) != 264 or struct.unpack_from("<HBB", data) != (264, 3, 0):
        raise ValueError(f"Unsupported GPU metrics layout: {path}")
    fclk = struct.unpack_from("<H", data, 182)[0]
    if fclk == 65535:
        raise ValueError("Fabric clock telemetry is unavailable")
    return {"average_fclk_MHz": fclk,
            "filter_us": struct.unpack_from("<I", data, 256)[0],
            "driver_snapshot_ns": struct.unpack_from("<Q", data, 104)[0]}


out = Path(sys.argv[1])
command = sys.argv[2:]
candidates = ([Path(os.environ["GPU_METRICS"])] if "GPU_METRICS" in os.environ else
              sorted(Path("/sys/class/drm").glob("card[0-9]*/device/gpu_metrics")))
supported = []
for candidate in candidates:
    try:
        read_metrics(candidate)
        supported.append(candidate)
    except (OSError, ValueError):
        pass
if len(supported) != 1:
    sys.exit("Set GPU_METRICS to a readable AMD APU gpu_metrics v3.0 file (expected one matching device).")
path = supported[0]
states = (path.parent / "pp_dpm_fclk").read_text()
(out / "fclk-states.txt").write_text(states)
print(f"Telemetry: {path}; clocks are firmware-filtered averages.", flush=True)
rows = []
start = time.monotonic()
with (out / "fabric-clock.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=["elapsed_s", "phase", "average_fclk_MHz", "filter_us", "driver_snapshot_ns"])
    writer.writeheader()

    def sample(phase):
        row = {"elapsed_s": round(time.monotonic() - start, 6), "phase": phase, **read_metrics(path)}
        rows.append(row)
        writer.writerow(row)
        time.sleep(0.1)

    for _ in range(10):
        sample("before")
    for run in range(1, 4):
        with (out / f"run-{run}-stdout.txt").open("wb") as stdout, (out / f"run-{run}-stderr.txt").open("wb") as stderr:
            with subprocess.Popen(command, stdout=stdout, stderr=stderr) as child:
                try:
                    while child.poll() is None:
                        sample(f"run-{run}")
                finally:
                    if child.poll() is None:
                        child.terminate()
                if child.wait() != 0:
                    sys.exit(f"Command failed: see run-{run}-stderr.txt")
    for _ in range(10):
        sample("after")

summary = {"metrics_path": str(path), "command": command, "phases": {}}
for phase in dict.fromkeys(r["phase"] for r in rows):
    values = [r["average_fclk_MHz"] for r in rows if r["phase"] == phase]
    stats = {"min_MHz": min(values), "median_MHz": statistics.median(values), "max_MHz": max(values), "samples": len(values)}
    summary["phases"][phase] = stats
    print(f"{phase}: {min(values)}–{max(values)} MHz; median {statistics.median(values):g} MHz")
summary["filter_us"] = sorted(set(r["filter_us"] for r in rows))
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Filter time constant: {summary['filter_us']} microseconds; timeline: {out / 'fabric-clock.csv'}")
