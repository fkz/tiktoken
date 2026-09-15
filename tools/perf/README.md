# Profiling `continue "Hello"`

Build once; compilation is excluded from all measurements:

```bash
zig build -Doptimize=ReleaseFast
tools/perf/memory.sh /path/to/gpt2-bf16.gguf Hello
tools/perf/cpu.sh /path/to/gpt2-bf16.gguf Hello
tools/perf/hotspots.sh /path/to/gpt2-bf16.gguf Hello
tools/perf/bandwidth.sh /path/to/gpt2-bf16.gguf Hello
tools/perf/bandwidth.sh --prefetch /path/to/gpt2-bf16.gguf Hello
tools/perf/fabric-clock.sh /path/to/gpt2-bf16.gguf Hello
# Synthetic check of memory bandwidth across physical CPU cores:
tools/perf/channel-check.sh --mt-s 5600
# CPU-only, GPU-only, and simultaneous streaming reads:
tools/perf/gpu-stream.sh
```

The prompt defaults to `Hello`. Use a model supported by the application (currently
GPT-2 BF16); these scripts do not add support for other architectures. Each script
runs `zig-out/bin/tiktoken continue MODEL PROMPT` from the project root. The current
application generates 200 tokens for `Hello`. Runs include loading, weight
reordering, prompt processing, generation, and teardown.

Requirements: Linux, Bash, Python 3 for memory sampling and bandwidth conversion, and Linux `perf` for CPU,
hotspot, and supported bandwidth measurements. On NixOS, for example:

```bash
nix shell nixpkgs#python3 nixpkgs#linuxPackages.perf
```

`BINARY`, `PYTHON`, and `PERF` can override executable paths. Each invocation creates
a unique ignored `perf-results/` directory with the command, binary SHA-256,
commit, kernel, date, stdout, and stderr. Generated text is on stderr in this app.
Scripts propagate failures; bandwidth exits 2 when unsupported. No script changes
kernel permissions, clears caches, installs tools, or invokes sudo.

## Memory: how much RAM does this run use?

`memory.sh` prints a summary and saves `summary.json` plus a `memory.csv` timeline
sampled roughly every 50 ms. It measures the application process, including its
threads; it does not aggregate any future subprocesses.

- **Peak RSS:** kernel-reported maximum resident physical memory via `getrusage`.
  This is the main starting point for sizing RAM for one process. Linux RSS
  accounting can differ slightly from the more detailed sampled RSS.
- **Rss / Pss:** sampled resident memory and resident memory with shared pages
  divided proportionally among processes. Model pages may be shared with other runs.
- **Pss_Anon / Pss_File:** anonymous memory (heap, reordered weights, caches) and
  file-backed memory (mapped model, executable, libraries).
- **Private_Dirty:** private modified resident pages; **Swap:** swapped-out pages.
- **VmSize:** virtual address space, including mappings and reservations. This
  is not a physical RAM requirement.
- Also reports CPU time, wall time, page faults, and context switches.

Sampled peaks can miss short spikes, and individual column peaks need not occur
at the same time; do not add them. Peak RSS includes resident mapped model pages,
so do not add the full model file size again. Allow additional RAM for the OS,
other applications, and workload variation. This measures observed usage, not the
minimum memory limit under which the command could finish. Sampling overhead and
up to one 50 ms polling interval affect wall time; use `cpu.sh` for timing.

The first invocation may generate `./tokens`. Later invocations also benefit
from filesystem cache. Run once to warm up, then repeat sequentially for a fair
comparison; keep the binary, model, prompt, and background load the same. Caches
are deliberately left alone. Record first-run and warm-run results separately.

## CPU and hotspots

`cpu.sh` saves `cpu.txt`: elapsed time, CPU time, scheduling, page faults,
instructions, cycles, and generic cache counters. Instructions/cycle can help
compare versions, but cache-misses alone cannot establish a DRAM bottleneck.
Hardware counters are user-space only; read perf's event availability and
multiplexing percentages before comparing results.

`hotspots.sh` saves a fresh `perf.data` and a text report in its results directory.
Open it interactively with `perf report -i perf-results/hotspots-.../perf.data`.
It samples user CPU cycles at 199 Hz with DWARF stacks; symbols are retained by
the project's build. Sampling adds overhead, so use separate timing runs.

## DRAM bandwidth

`bandwidth.sh` automatically chooses an available measurement. It saves raw
100 ms interval counts in `bandwidth.txt`, bandwidth in `rates.txt`, and event
definitions/scales in `events.txt`. Rates use actual timestamp differences,
including the final partial interval; the average is total bytes / total time.
GB/s is decimal; GiB/s uses 2^30 bytes. Counters running for less than 99% of
enabled time are rejected to avoid reporting heavily multiplexed measurements.

### Ryzen: DRAM-to-L2 read bandwidth estimate

This **works on the Ryzen AI 5 340 here without root**. It uses the named perf
events `l2_fill_rsp_src.dram_io_near` and `l2_fill_rsp_src.dram_io_far`, which
identify L2 fills supplied by DRAM/MMIO rather than other caches. See the
[Linux Zen 5 event definitions](https://github.com/torvalds/linux/blob/master/tools/perf/pmu-events/arch/x86/amdzen5/l2-cache.json).
The script checks the cache line size reported by sysfs is 64 bytes and computes:

```text
estimated read bandwidth = (near fills + far fills) × 64 bytes / elapsed seconds
```

Counters follow the command and its threads in user space. The report includes
prefetched lines that fill L2, so no separate prefetch counter is added. This is
an estimate of incoming DRAM traffic observed at the CPU cache, not a measurement
of all memory-controller traffic: it excludes writebacks, kernel work, and GPU
traffic. The event also admits MMIO; interpreting each fill as a 64-byte RAM line
assumes ordinary cacheable memory, as used by this inference command. In-flight
prefetches at scheduling boundaries can affect attribution. No generic cache-miss
counter, disk I/O counter, or assumed model-byte count is used.

### Exposed Intel memory controllers: system-wide reads and writes

Where available, `uncore_imc` read/write CAS events take precedence. Kernel scales
convert events to bytes; traffic is summed across controllers. These counters
cover the **whole system**, including background activity, and usually require
additional perf privileges. The script reports errors without changing system
policy. Machines exposing neither method get an explicit unsupported error.

### Prefetch versus demand fills (AMD)

```bash
# All L1 data-cache fills, including lines supplied by L2/L3:
tools/perf/bandwidth.sh --prefetch /path/to/gpt2-bf16.gguf Hello
# Only L1 fills whose reported source is DRAM/MMIO:
tools/perf/bandwidth.sh --prefetch --dram /path/to/gpt2-bf16.gguf Hello
# DRAM-to-L2 fills, including lines never fetched into L1, plus prefetch activity:
tools/perf/bandwidth.sh --prefetch --l2 /path/to/gpt2-bf16.gguf Hello
```

These invoke `prefetch.sh`, which can also be called directly with the same
arguments after `--prefetch`. The default and `--dram` views separate **Demand**, **Hardware PF**,
and **Software PF** fills using `ls_dmnd_fills_from_sys`, `ls_hw_pf_dc_fills`,
and `ls_sw_pf_dc_fills` respectively. The suffix is `.all` by default or
`.dram_io_all` with `--dram`; see the
[Linux Zen 5 event definitions](https://github.com/torvalds/linux/blob/master/tools/perf/pmu-events/arch/x86/amdzen5/load-store.json).
These L1 views run three counters together for the command and its threads in user space.
Unsupported events, unavailable counts, or coverage below 99% cause an error.

The report shows 100 ms intervals, total fills, average byte-equivalent GiB/s
(fills × the verified 64-byte cache line size / wall time), and each category's
share of observed fills. Timing includes setup and teardown. Results are saved
under `perf-results/prefetch-*/`, including raw `prefetch.txt`, `rates.txt`,
event definitions, the command, and its binary hash.

These are **L1 fill categories**, not a complete split of DRAM bandwidth or of
all loads. An L1 hit causes no fill. A demand fill from L2 may have been
prefetched into L2 earlier. Direct L2 hardware prefetch activity is not counted
as an L1 prefetch fill, and the DRAM-only view excludes fills supplied by caches.
The percentages therefore do not measure prefetch accuracy or the fraction of
loads whose latency was hidden. Use ordinary `bandwidth.sh` for DRAM-to-L2 rates.

To compare software prefetching, rebuild with the same optimization and
thread count, changing only `-Dprefetch=0` versus `-Dprefetch=1`, and run
the same command sequentially. Hardware prefetching still operates when the
software option is disabled. Compare elapsed time as well as fill categories;
more software-prefetch fills alone do not establish a speedup.

The `--l2` view addresses prefetches that stop before L1. It runs four counters
simultaneously and reports three independent measurements:

| Column | Measurement |
| --- | --- |
| `DRAM_L2_GiB/s` | Near + far DRAM/MMIO-sourced L2 fills × 64 / elapsed time, including prefetched lines never consumed by L1 |
| `HW_PF_Mreq/s` | Millions of L1/L2 hardware-prefetch requests per second that miss both L2 and L3 |
| `SW_PF_L1_Mfill/s` | Millions of software-prefetch L1 fills per second whose reported source is DRAM/MMIO |

The hardware event is `l2_pf_miss_l2_l3.l1_dc_l2_hwpf`; its definition describes
requests accepted by the L2 pipeline, without identifying the eventual data
source or counting completed fills. See the
[L2 event definitions](https://github.com/torvalds/linux/blob/master/tools/perf/pmu-events/arch/x86/amdzen5/l2-cache.json).
The software event remains `ls_sw_pf_dc_fills.dram_io_all`, with its L1 scope.
Consequently these columns must not be added or subtracted to construct a
prefetched/non-prefetched DRAM split. No such percentage is reported.
The DRAM-to-L2 estimate also excludes L3-only fills, writebacks, kernel and GPU
traffic. `--dram` and `--l2` select alternative views; they are not combined.

### Comparing prefetch distances and build modes

```bash
python3 tools/perf/prefetch-timing.py /path/to/gpt2-bf16.gguf \
  --prompt 'Why are you so' --rounds 30
```

This builds all 12 combinations of `ReleaseSmall`, `ReleaseFast`, `ReleaseSafe`
and `-Dprefetch=0,1,2,3` from a frozen source snapshot. Zero disables software
prefetching; 1/2/3 look ahead 4/8/12 KiB in the packed weight stream. The run
uses the build's default one compute thread and inherits CPU affinity; it does
not change power settings or pin the process to a particular core.

Compilation completes before measurements begin. After one warm-up per variant,
each randomized round runs all 12 variants once, sequentially. Whole-command
wall time includes model loading, weight reordering, prompt processing, 200
generated tokens, and teardown. Output is saved to files instead of displayed
in a terminal. An existing tokenizer cache is copied to the experiment's runtime
directory; caches are not cleared between runs. Failed commands and changing
output within a variant stop the experiment. No timing samples are discarded.

Each `perf-results/prefetch-timing-*/` directory saves source and binary hashes,
build logs, commands, environment, randomized order, warm-ups, per-run wall/user/
system times and resource counts, generated output, `timings.csv`, `summary.json`,
all 66 comparisons in `pairwise.json`, and a readable `report.md`.

The report gives medians and distribution-free 95% median confidence intervals.
Comparisons pair timings from the same round and use exact two-sided sign tests,
with Holm correction across all 66 pairs at family-wise alpha 0.05. A difference
is reported as significant only if its adjusted p-value is below 0.05. Small
differences may remain unresolved; the results apply to this workload and machine
state. `--rounds` and `--seed` set the experiment size and randomized schedule.
Use `--analyze RESULTS_DIR` to regenerate the analysis from saved timings.

## Example baseline (2026-09-15)

Ryzen AI 5 340, ReleaseFast, `gpt2-bf16.gguf`, `Hello`, existing tokenizer cache:

| Measurement | Observed |
| --- | ---: |
| Kernel peak RSS (warm repeat) | 621.1 MiB |
| Sampled resident memory | 622.1 MiB |
| Sampled anonymous / file-backed PSS | 383.2 / 238.9 MiB |
| Sampled virtual address space | 823.5 MiB |
| Sampled swap | 0 MiB |
| Separate CPU-counter run elapsed | 1.681 s |
| Instructions / cycle | 0.79 |
| DRAM-to-L2 read estimate (separate run) | 42.667 GB/s / 39.737 GiB/s |
| Read traffic / duration in bandwidth run | 48.235 GiB / 1.214 s |
| Full memory-controller read + write traffic | Unavailable on exposed PMUs |

The first memory run had 385 major page faults; the warm repeat had zero.
The separate hotspot run collected 612 samples with no lost samples; the output
projection `BiasWeightCalc(1,768,50304,false).calculate` accounted for about 19%.
These are individual observations, not statistical benchmark averages.
The bandwidth run's busiest 100 ms interval was 46.543 GiB/s; startup and teardown
are included in its overall average. All its counters reported 100% enabled-time
coverage. Timing differs across these separately collected runs.

## Checking usable dual-channel bandwidth

`channel-check.sh --mt-s 5600` builds a small AVX2 read-only streaming benchmark
with Zig's C compiler, then uses Python 3 to test 1, 2, 4, and all physical cores
(up to 64). It needs Linux/x86 with AVX2 and about 1 GiB of available RAM. Pass
the **configured** transfer rate from firmware/SMBIOS, not a module's advertised
maximum. This command does not take a model or modify the inference executable.

The user-provided SMBIOS report for this machine shows two 48 GiB, 64-bit DDR5
SODIMMs on `P0 CHANNEL A` and `P0 CHANNEL B`, both configured at 5600 MT/s.
That gives theoretical data rates of 44.8 GB/s for one 64-bit channel and
89.6 GB/s for two. DDR5's internal subchannels are already included in each
module's 64-bit data width; do not double-count them.

The test partitions a fixed 1 GiB buffer into disjoint ranges and reads it 32
times. Every read contributes to a checked sum, and a compiler memory barrier
prevents reusing loads across passes. Initialization, allocation, and worker
creation are outside the timed interval. The buffer is much larger than the
machine's 16 MiB L3 cache. Workers are pinned to separate physical cores, with
the higher-frequency cores selected first. After a warm-up, three repeats per
worker count run in shuffled order. This measures useful read bytes per second;
it does not include stores or read-for-ownership traffic in its numerator.

Observed results on this machine (2026-09-15):

| CPU workers | Median GB/s | Range GB/s |
| ---: | ---: | ---: |
| 1 | 52.61 | 50.35–54.12 |
| 2 | 58.30 | 58.02–58.66 |
| 4 | 57.12 | 56.42–58.42 |
| 6 | 56.72 | 55.56–56.90 |

The best median is **1.30× one channel's theoretical maximum**, demonstrating
usable bandwidth beyond a single channel. One CPU worker can already use both
memory channels; worker count is not channel count. Two workers improve this
particular streaming test by 11% over one worker.

A separate two-worker cross-check read exactly 32 GiB at 57.96 GB/s. The
DRAM-to-L2 counters reported 32.071 GiB for the full process, with steady intervals
around 55 GiB/s (59 GB/s), consistent with the timed streaming result. The counter
average includes initialization/teardown, which the streaming timer excludes.

Thus inference's roughly 45 GB/s is about 77% of the observed streaming capacity,
even though it is only about half the theoretical two-channel maximum. These
measurements do not isolate why it falls below the theoretical maximum. Exact
single-channel versus dual-channel inference speedup requires rerunning the same
command with one channel disabled or one module removed; this test makes no
firmware or hardware changes. A result below one channel's ceiling would be
inconclusive, rather than proof that only one channel is active.

Each run saves CPU selection, memory speed, individual results, and medians to
`perf-results/channel-check-*/`. Set `PYTHON` or `ZIG` to override their paths.

## Fabric clock under load

`fabric-clock.sh` samples AMD APU `gpu_metrics` telemetry every 100 ms before,
during three consecutive command runs, and afterwards. It saves
`fabric-clock.csv`, `summary.json`, and the available FCLK states. This requires
Python 3 and the 264-byte version 3.0 metrics layout; unknown layouts are rejected.
Set `GPU_METRICS` to select a sysfs metrics file if multiple supported GPUs exist.

The [Linux metrics definition](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/include/kgd_pp_interface.h)
defines `average_fclk_frequency` in MHz. This is a firmware-filtered value; the
filter time constant on this machine is **one second**. Short bursts and gaps
between commands therefore affect the readings. `pp_dpm_fclk` lists available
states but does not mark an active state here, so its highest entry alone is not
a live clock measurement.

During three `Hello` runs, telemetry rose from a pre-run median of 824.5 MHz to a
third-run median of 1878 MHz and maximum of 1915 MHz. The highest listed state is
1960 MHz. The clock ramps up under load and drops afterwards; these measurements
do not establish a constant instantaneous 1960 MHz clock.

## GPU and simultaneous CPU/GPU streaming

`gpu-stream.sh` builds and runs a Vulkan compute shader on an AMD integrated GPU.
It refuses software Vulkan devices. Requirements: Linux/x86 with AVX2, GPU render
device access, Zig, Python 3, glslangValidator, Vulkan headers/loader, and
pkg-config. On NixOS:

```bash
nix-shell -p zig python3 glslang pkg-config vulkan-headers vulkan-loader \
  --run 'tools/perf/gpu-stream.sh'
```

The default is three 3-second measurements per case in shuffled order. Use
`--seconds 5 --repeats 3` to change the sampling. The test needs roughly 2 GiB of
buffer memory when both processors run. GPU access must be available in the
execution environment; no root privileges or graphics configuration changes are
required on a normal desktop session.

Both processors read separate 1 GiB buffers. GPU accesses are coalesced and every
input vector contributes to a checked output sum. Each GPU dispatch reads the
entire input and writes 1 MiB of checksums; the reported read rate excludes those
small writes. Input allocation, initialization, expected-checksum calculation,
shader creation, and GPU warm-up are outside the measured interval. CPU workers
use the two fastest allowed physical cores. Both processes finish setup before
waiting for the same absolute start time.

The primary GPU rate uses wall time, including dispatch and fence overhead.
`gpu_GB_s` in individual results additionally reports shader throughput from
[Vulkan timestamps](https://docs.vulkan.org/spec/latest/chapters/queries.html).
For simultaneous runs, total completed CPU and GPU read bytes are divided by
their common wall-clock span. The script rejects less than 95% interval overlap;
the observed overlap here was above 99%. It does not add standalone results or
GPU timestamp rates to produce the combined number. This measures useful read
throughput, not all physical DRAM traffic, and does not measure inference speed.

`perf-results/gpu-stream-*/` contains raw runs, medians, device/driver logs, CPU
selection, shader/binary hashes, and the compiled benchmark. `ZIG`, `PYTHON`, and
`GLSLANG` override tool paths. `VULKAN_INCLUDE` and `VULKAN_LIB` can replace
pkg-config discovery. `VK_DRIVER_FILES` can select the Radeon ICD explicitly.

The shared CPU reader still defaults to 32 passes for `channel-check.sh`.
`STREAM_SECONDS` enables duration-based runs, while `STREAM_SYNC` enables the
READY/absolute-start protocol used by the GPU comparison driver. These variables
are normally set by the driver, not by the user.

### Observed Radeon 840M results (2026-09-15)

RADV KRACKAN1, Mesa 26.1.8, DDR5-5600 on both channels; three valid 3-second
measurements per case, 1 GiB per processor:

| Case | Median sustained reads | Range |
| --- | ---: | ---: |
| CPU alone, two workers | 58.45 GB/s | 57.00–59.59 GB/s |
| GPU alone | 79.05 GB/s | 77.45–79.67 GB/s |
| CPU and GPU together | 78.92 GB/s | 78.75–79.05 GB/s |

In the simultaneous runs, individual medians were 20.04 GB/s on the CPU and
59.08 GB/s on the GPU. Their individual timing windows differ slightly; the
combined result above uses the common span. GPU output checksums passed before
and after every run, and CPU checksums passed. One CPU-only sample overlapped a
CPU regression check, was explicitly marked excluded in the raw results, and
was replaced with a clean sample.

The GPU achieves about 88% of the memory bus's 89.6 GB/s theoretical peak.
Adding CPU readers does not improve aggregate throughput in this test: they
compete for the same RAM bandwidth. The GPU can use bandwidth beyond what the
CPU-only test reaches, but this does not by itself establish an inference speedup.
