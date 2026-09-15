#!/usr/bin/env python3
"""Linux process memory sampling plus kernel-reported peak RSS, no dependencies."""
import csv
import json
import pathlib
import resource
import subprocess
import sys
import time


def fields(path):
    result = {}
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            parts = value.split()
            if parts and parts[0].isdigit():
                result[key] = int(parts[0])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


def main():
    out = pathlib.Path(sys.argv[1])
    command = sys.argv[2:]
    names = ["Rss", "Pss", "Pss_Anon", "Pss_File", "Private_Dirty", "Swap", "VmSize"]
    peaks = dict.fromkeys(names, 0)
    seen = set()
    samples = 0
    start = time.monotonic()
    with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        proc = pathlib.Path(f"/proc/{child.pid}")
        try:
            with (out / "memory.csv").open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(["elapsed_s"] + [name + "_KiB" for name in names])
                while child.poll() is None:
                    values = fields(proc / "smaps_rollup")
                    values.update({k: v for k, v in fields(proc / "status").items() if k == "VmSize"})
                    seen.update(values)
                    writer.writerow([round(time.monotonic() - start, 6)] + [values.get(n, "") for n in names])
                    for name in names:
                        peaks[name] = max(peaks[name], values.get(name, 0))
                    samples += 1
                    time.sleep(0.05)
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait()
    elapsed = time.monotonic() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "command": command, "exit_code": child.returncode, "wall_seconds": elapsed,
        "user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime,
        "peak_rss_KiB": usage.ru_maxrss,
        "sampled_peaks_KiB": {n: peaks[n] if n in seen else None for n in names},
        "minor_faults": usage.ru_minflt, "major_faults": usage.ru_majflt,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw, "samples": samples,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wall: {elapsed:.3f} s; CPU: {usage.ru_utime + usage.ru_stime:.3f} s")
    print(f"Peak resident RAM (kernel): {usage.ru_maxrss / 1024:.1f} MiB")
    for name in names:
        if name in seen:
            print(f"Sampled peak {name}: {peaks[name] / 1024:.1f} MiB")
    if "Pss" not in seen:
        print("Detailed /proc memory accounting unavailable; use kernel peak RSS above.")
    print(f"Page faults: {usage.ru_minflt} minor, {usage.ru_majflt} major")
    print(f"Command exit code: {child.returncode}; generated text: {out / 'stderr.txt'}")
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    sys.exit(main())
