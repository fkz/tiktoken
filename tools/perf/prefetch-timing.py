#!/usr/bin/env python3
"""Build a frozen source snapshot and compare 12 variants in randomized blocks."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import tempfile
import time


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def median_ci(values, alpha=0.05):
    """Conservative exact binomial/order-statistic CI for a population median."""
    ordered = sorted(values)
    n = len(ordered)
    k = 0
    for candidate in range(1, n // 2 + 1):
        tail = sum(math.comb(n, j) for j in range(candidate)) / 2**n
        if 2 * tail <= alpha:
            k = candidate
    return [ordered[k - 1], ordered[n - k]] if k else [None, None]


def sign_p(a, b):
    wins = sum(x < y for x, y in zip(a, b))
    losses = sum(x > y for x, y in zip(a, b))
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(wins, losses) + 1)) / 2**n)


def analyze(out):
    records = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    metadata = json.loads((out / "environment.json").read_text())
    measured = [r for r in records if r["phase"] == "measurement"]
    grouped = {}
    for r in measured:
        grouped.setdefault(r["variant"], []).append(r)
    if len(grouped) != 12 or any(len(v) != metadata["rounds"] for v in grouped.values()):
        raise RuntimeError("Incomplete experiment; refusing significance analysis")
    series = {}
    summary = []
    for name, rows in grouped.items():
        rows.sort(key=lambda r: r["round"])
        values = [r["wall_s"] for r in rows]
        series[name] = values
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        summary.append({"variant": name, "n": len(values), "median_s": statistics.median(values),
                        "median_95ci_s": median_ci(values), "mean_s": statistics.mean(values),
                        "stdev_s": statistics.stdev(values), "q1_s": quartiles[0], "q3_s": quartiles[2],
                        "min_s": min(values), "max_s": max(values),
                        "median_user_s": statistics.median(r["user_s"] for r in rows),
                        "median_sys_s": statistics.median(r["sys_s"] for r in rows)})
    pairs = []
    for a, b in itertools.combinations(sorted(series), 2):
        ratios = [x / y for x, y in zip(series[a], series[b])]
        pairs.append({"a": a, "b": b, "median_paired_time_ratio_a_over_b": statistics.median(ratios),
                      "ratio_95ci": median_ci(ratios), "p_sign": sign_p(series[a], series[b])})
    # Correct all 66 comparisons, including selection of the observed winner.
    adjusted = 0.0
    for rank, pair in enumerate(sorted(pairs, key=lambda p: p["p_sign"])):
        adjusted = max(adjusted, min(1.0, (len(pairs) - rank) * pair["p_sign"]))
        pair["p_holm"] = adjusted
    summary.sort(key=lambda r: r["median_s"])
    best = summary[0]["variant"]
    for row in summary:
        if row["variant"] == best:
            row["paired_slowdown_vs_best_pct"] = 0.0
            row["p_holm_vs_best"] = 1.0
            continue
        pair = next(p for p in pairs if {p["a"], p["b"]} == {row["variant"], best})
        ratios = [x / y for x, y in zip(series[row["variant"]], series[best])]
        row["paired_slowdown_vs_best_pct"] = (statistics.median(ratios) - 1) * 100
        row["paired_slowdown_95ci_pct"] = [(x - 1) * 100 for x in median_ci(ratios)]
        row["p_holm_vs_best"] = pair["p_holm"]
    write_json(out / "summary.json", summary)
    write_json(out / "pairwise.json", pairs)
    with (out / "timings.csv").open("w", newline="") as f:
        columns = ["variant", "round", "position", "wall_s", "user_s", "sys_s", "minor_faults",
                   "major_faults", "voluntary_switches", "involuntary_switches", "output_sha256"]
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measured)
    lines = ["# Prefetch and optimization timing comparison", "",
             f"Prompt: `{metadata['prompt']}`. {metadata['rounds']} randomized complete blocks; "
             "one warm-up per variant; one compute thread; inherited CPU affinity.", "",
             "Whole-command wall time includes model loading, weight reordering, prompt processing, "
             "200 generated tokens, and teardown. Compilation is excluded. Model/tokenizer caches are warm. "
             "Output is captured to files rather than rendered in a terminal.", "",
             "| Variant | Median s | 95% median CI | IQR s | Paired slowdown vs best | Holm p vs best |",
             "| --- | ---: | --- | --- | ---: | ---: |"]
    for row in summary:
        lo, hi = row["median_95ci_s"]
        lines.append(f"| {row['variant']} | {row['median_s']:.4f} | {lo:.4f}–{hi:.4f} | "
                     f"{row['q1_s']:.4f}–{row['q3_s']:.4f} | {row['paired_slowdown_vs_best_pct']:+.2f}% | "
                     f"{row['p_holm_vs_best']:.5g} |")
    lines.extend(["", "Intervals are marginal, distribution-free 95% confidence intervals for the median. "
                  "Paired slowdowns compare configurations within each randomized round. Two-sided exact sign "
                  "tests assess paired timing differences; Holm adjustment covers all 66 pairwise comparisons "
                  "at family-wise alpha 0.05. Unadjusted intervals and adjusted p-values answer different questions.", "",
                  "No timing samples were removed. These results apply to this prompt, model, thread count, "
                  "and machine state; they do not establish general inference performance. Repeated rounds "
                  "reduce drift effects but cannot eliminate correlated background or thermal variation.", "",
                  "All runs exited successfully and produced the same bytes as their variant's warm-up. "
                  f"Distinct output hashes across variants: {len(set(r['output_sha256'] for r in measured))}.", ""])
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--prompt", default="Why are you so")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analyze", type=Path)
    args = parser.parse_args()
    if args.analyze:
        analyze(args.analyze.resolve())
        return
    if args.model is None or not args.model.is_file() or args.rounds < 10:
        parser.error("Provide a model and at least 10 rounds")
    root = Path(__file__).resolve().parents[2]
    out = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="prefetch-timing-", dir=root / "perf-results"))
    out.mkdir(parents=True, exist_ok=True)
    if (out / "runs.jsonl").exists():
        parser.error("Output already contains a run; use a fresh directory")
    source = out / "source"
    source.mkdir()
    files = [root / "build.zig", root / "build.zig.zon", *sorted((root / "src").rglob("*.zig")),
             *sorted((root / "tools").rglob("*.zig"))]
    for path in files:
        dest = source / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    # Separate runtime cwd keeps the tokenizer cache and source immutable during the experiment.
    runtime = out / "runtime"
    runtime.mkdir()
    if (root / "tokens").exists():
        shutil.copy2(root / "tokens", runtime / "tokens")
    zig = shutil.which(os.environ.get("ZIG", "zig"))
    if not zig:
        parser.error("Zig is required")
    configs = [{"variant": f"{opt}-p{pf}", "optimize": opt, "prefetch": pf}
               for opt in ["ReleaseSmall", "ReleaseFast", "ReleaseSafe"] for pf in range(4)]
    rng = random.Random(args.seed)
    schedule = []
    for round_number in range(args.rounds):
        names = [c["variant"] for c in configs]
        rng.shuffle(names)
        schedule.append(names)
    metadata = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(),
                "zig": zig, "zig_version": subprocess.check_output([zig, "version"], text=True).strip(),
                "model": str(args.model.resolve()), "model_sha256": digest(args.model), "prompt": args.prompt,
                "rounds": args.rounds, "seed": args.seed, "thread_count": 1,
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "initial_loadavg": os.getloadavg(),
                "source_sha256": {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()},
                "configs": configs, "schedule": schedule,
                "statistics": "30 planned blocks by default; exact paired sign tests, Holm correction over 66 pairs; no outlier exclusions"}
    write_json(out / "environment.json", metadata)
    (out / "lscpu.txt").write_text(subprocess.check_output(["lscpu"], text=True))
    (out / "working.diff").write_bytes(subprocess.check_output(["git", "diff"], cwd=root))
    shutil.copy2(__file__, out / "runner.py")
    print(f"Results: {out}", flush=True)
    binaries = {}
    for config in configs:
        name = config["variant"]
        prefix = out / "builds" / name
        prefix.mkdir(parents=True)
        command = [zig, "build", f"-Doptimize={config['optimize']}", f"-Dprefetch={config['prefetch']}",
                   "--prefix", str(prefix), "--global-cache-dir", str(root / ".zig-cache" / "perf-global")]
        print(f"Building {name}", flush=True)
        started = time.monotonic()
        with (prefix / "build.log").open("w") as log:
            subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = prefix / "bin" / "tiktoken"
        binaries[name] = binary
        write_json(prefix / "build.json", {"command": command, "seconds": time.monotonic() - started,
                                          "binary_sha256": digest(binary), "binary_bytes": binary.stat().st_size})
    reference = {}
    samples = out / "samples"
    samples.mkdir()
    def measure(name, phase, round_number, position):
        stem = f"{phase}-{round_number:02}-{position:02}-{name}"
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        load = os.getloadavg()
        with (samples / (stem + ".stdout")).open("wb") as stdout, (samples / (stem + ".stderr")).open("wb") as stderr:
            started = time.perf_counter_ns()
            result = subprocess.run([str(binaries[name]), "continue", str(args.model.resolve()), args.prompt],
                                    cwd=runtime, stdout=stdout, stderr=stderr)
            elapsed = (time.perf_counter_ns() - started) / 1e9
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        output_hash = hashlib.sha256((samples / (stem + ".stdout")).read_bytes() + b"\0" +
                                     (samples / (stem + ".stderr")).read_bytes()).hexdigest()
        record = {"variant": name, "phase": phase, "round": round_number, "position": position,
                  "wall_s": elapsed, "user_s": after.ru_utime - before.ru_utime,
                  "sys_s": after.ru_stime - before.ru_stime, "minor_faults": after.ru_minflt - before.ru_minflt,
                  "major_faults": after.ru_majflt - before.ru_majflt,
                  "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
                  "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
                  "loadavg_before": load, "returncode": result.returncode, "output_sha256": output_hash}
        with (out / "runs.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); see {samples / (stem + '.stderr')}")
        if phase == "warmup":
            reference[name] = output_hash
        elif output_hash != reference[name]:
            raise RuntimeError(f"Output changed between runs of {name}; inspect samples before comparing timings")
        return elapsed
    print("Builds complete. Warming every variant before timed rounds.", flush=True)
    for position, name in enumerate(schedule[0]):
        elapsed = measure(name, "warmup", 0, position)
        print(f"Warm-up {name}: {elapsed:.3f}s", flush=True)
    print(f"Starting {args.rounds} randomized rounds ({args.rounds * 12} measurements).", flush=True)
    for round_number, names in enumerate(schedule, 1):
        times = [measure(name, "measurement", round_number, position) for position, name in enumerate(names)]
        print(f"Round {round_number}/{args.rounds}: {sum(times):.1f}s total; "
              f"range {min(times):.3f}–{max(times):.3f}s", flush=True)
    analyze(out)


if __name__ == "__main__":
    main()
