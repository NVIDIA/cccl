#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
import sys
from io import StringIO
from pathlib import Path


# Maps each NVBench binary to (schema metric name, allowed Skipped-row count).
# Each binary contributes one metric (the geomean of its non-skipped positive
# GlobalMem BW rows, in GiB/s) instead of folding into a single aggregate.
#
# `allowed_skips` is the number of cells nvbench is permitted to mark
# `Skipped: Yes` for a LEGITIMATE reason (the multi-channel Elements=1G cells
# whose row stride overflows OffsetT use `state.skip(...)`; single-channel
# binaries have no legitimate skips). A correctness-check failure now aborts the
# binary (non-zero exit, caught below), so it can never reach this filter -- but
# we additionally HARD-FAIL if the observed skip count exceeds the allowance, so
# no future change can quietly drop hard cells from the geomean to inflate it.
# Multi-channel legitimate skips = Elements(1G:1) * Bins(3) * SampleT(2) *
# InputShape(6) = 36 cells whose row stride (elements * num_channels) overflows
# OffsetT. Single-channel binaries have no legitimate skips. See project-layout.md.
MULTI_ALLOWED_SKIPS = 36

BENCHMARKS = [
    ("histogram_even", "cub.bench.histogram.even.base", 0),
    ("histogram_range", "cub.bench.histogram.range.base", 0),
    ("histogram_multi_even", "cub.bench.histogram.multi.even.base", MULTI_ALLOWED_SKIPS),
    ("histogram_multi_range", "cub.bench.histogram.multi.range.base", MULTI_ALLOWED_SKIPS),
]

# Axis set per project-layout.md: high-bin regime only (>16K), and the tunable
# InputShape axis (name[:knob]) replacing the old bitwise-AND Entropy axis.
# concentrated:E spans uniform(E=1.0)..constant(E=0.0); powerlaw is the
# cache-vs-atomics discriminator; capacity_cliff, stale_resident, and
# hash_synonym are the three cache adversaries (treated as floors, not maximize
# targets) attacking capacity / eviction / hash-collision respectively.
AXES = [
    ("SampleT{ct}", "[I32,F64]"),
    ("Elements{io}", "[1048576,16777216,67108864,268435456,1073741824]"),
    ("Bins", "[65536,262144,1048576]"),
    # NOTE: capacity_cliff was dropped from histogram_inputs.cuh on this branch
    # (commit 0cf4594ba6); the stock helper's axis list is stale and would skip
    # 30 cells -> trip the allowed_skips=0 guard. Replaced with sawtooth (an
    # implemented shape) so the metric-neutrality run completes. Local copy for
    # the cached-privatized experiment only.
    ("InputShape", "[concentrated:1.0,concentrated:0.0,powerlaw:0.5,sawtooth,stale_resident,hash_synonym]"),
]


def run_one(binary: Path, allowed_skips: int, args: argparse.Namespace) -> list[float]:
    cmd = [
        str(binary),
        "--benchmark",
        "base",
    ]
    for name, value in AXES:
        cmd += ["--axis", f"{name}={value}"]
    cmd += [
        "--min-samples",
        str(args.min_samples),
        "--min-time",
        str(args.min_time),
        "--timeout",
        str(args.timeout),
        "--csv",
        "stdout",
        "--quiet",
    ]
    print(f"[autocuda histogram] running {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stdout, end="")
        raise SystemExit(proc.returncode)

    rows = list(csv.DictReader(StringIO(proc.stdout)))
    values = []
    skipped = 0
    for row in rows:
        if row.get("Skipped") == "Yes":
            skipped += 1
            continue
        raw = row.get("GlobalMem BW (bytes/sec)", "")
        if not raw:
            continue
        value = float(raw) / (1024.0**3)
        if value > 0.0 and math.isfinite(value):
            values.append(value)
    if not values:
        raise RuntimeError(f"{binary.name} produced no positive non-skipped bandwidth rows")

    # Reward-hacking guard: a kernel that computes wrong counts now aborts the
    # binary (non-zero exit, handled above). As defense in depth, refuse to
    # report a metric if MORE cells were skipped than the legitimate allowance,
    # since extra skips would silently shrink the cell set and inflate the geomean.
    if skipped > allowed_skips:
        # Dump the raw CSV to stderr (not stdout) so it can't bury the final
        # metric summary that callers parse from stdout.
        print(proc.stdout, file=sys.stderr, end="")
        raise RuntimeError(
            f"{binary.name}: {skipped} cells Skipped but only {allowed_skips} legitimate "
            f"skips are allowed; refusing to report a metric over a reduced cell set "
            f"(a kernel correctness failure or an unexpected skip is the likely cause)"
        )

    geo = math.exp(sum(math.log(v) for v in values) / len(values))
    # Per-binary diagnostics go to STDERR so stdout carries only the final,
    # compact metric summary (printed once at the end of main()). This keeps the
    # four `histogram_*=` lines together and last, so a caller that pipes through
    # `tail` cannot capture a truncated/partial result.
    print(f"{binary.name}: rows={len(values)} skipped={skipped} geomean_gib_per_sec={geo:.6f}", file=sys.stderr)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-time", default="0.01")
    parser.add_argument("--timeout", default="10")
    args = parser.parse_args()

    root = Path.cwd()
    # Compute every metric first (diagnostics stream to stderr from run_one), then
    # emit ONE compact summary block to stdout at the very end. The four
    # `histogram_*=` lines are always the final lines of stdout and are fenced by
    # explicit BEGIN/END markers, so a caller that pipes through `tail` (or only
    # reads the tail of a captured log) always sees the complete, untruncated set.
    metrics: list[tuple[str, float]] = []
    for metric_name, binary, allowed_skips in BENCHMARKS:
        values = run_one(root / "build/autocuda/cub-benchmark/bin" / binary, allowed_skips, args)
        geo = math.exp(sum(math.log(v) for v in values) / len(values))
        metrics.append((metric_name, geo))

    print("===HISTOGRAM_RESULTS_BEGIN===")
    for metric_name, geo in metrics:
        print(f"{metric_name}={geo:.6f}")
    print("===HISTOGRAM_RESULTS_END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
