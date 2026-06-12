#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""64-bit-counter, very-large-input histogram characterization sweep.

Runs the 64-bit-counter / 64-bit-offset bench variant (cub.bench.histogram.<bin>.base.u64,
built by build_u64_variants.sh with CounterT=unsigned long long, OffsetT=long long) across
bins x VERY LARGE element counts (up to 32-64 GiB of samples). A 64-bit OffsetT is required
for >2^31 elements; a 64-bit counter (unsigned long long -- uint64_t has no CUDA atomicAdd
overload) is what a histogram of tens of billions of samples needs so per-bin counts do not
overflow a 32-bit counter.

Two things vary that the default (I32-counter, <=256M-element) sweeps cannot show:
  - counter width: an 8-byte counter halves the on-chip dynamic-SMEM byte budget, so the
    selector's on-chip bin cap is ~half (B200 single-channel ~28672 bins vs ~57344 for I32).
  - input scale: 1-8 G elements (4-64 GiB), where atomic throughput / amortization dominate.

For each (binary, SampleT, Elements, Bins, InputShape) cell it records the selector's
`default` GiB/s plus the `_selected` launch tag (so the plotter labels which algorithm ran
at each 64-bit-counter cell). No upstream-`main` baseline: main's bench is I32-counter only,
so a 64-bit-counter-vs-32-bit-counter-main comparison would be apples-to-oranges. The story
here is which branch algorithm wins per (bins, N) at 64-bit width and scale, and how the
halved on-chip cap shifts the smem_privatized -> high-bin crossover.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
from io import StringIO
from pathlib import Path

BINARIES = {
    "even": "cub.bench.histogram.even.base.u64",
    "range": "cub.bench.histogram.range.base.u64",
}

# Large-input element counts. F64 (8B) * 8G = 64 GiB; I32 (4B) * 8G = 32 GiB. All > 2^31,
# exercising the 64-bit OffsetT. Smaller anchors included so the large-N trend is visible.
DEFAULT_ELEMENTS = [1 << 28, 1 << 30, 1 << 31, 1 << 32, 1 << 33]  # 256M, 1G, 2G, 4G, 8G
# Bins spanning the on-chip tier (now ~half due to 8B counter), the gather/hybrid tier,
# and the direct-atomic tiers.
DEFAULT_BINS = [256, 4096, 16384, 32768, 65536, 262144, 1048576]
DEFAULT_SAMPLES = ["I32", "F64"]
DEFAULT_SHAPES = [
    "concentrated:1.0", "concentrated:0.5", "powerlaw:0.5", "zipf:1.0",
    "hash_synonym", "strided_sweep", "sawtooth",
]

_LAUNCH_RE = re.compile(r"\[launch\] bins=(\d+) ch=(\d+) ran=([a-z_:]+)")


def ran_tag(stderr: str):
    rans = {m.group(3) for m in _LAUNCH_RE.finditer(stderr)}
    return next(iter(rans)) if len(rans) == 1 else None


def cell_key(sample, elements, bins, shape):
    return f"{sample}|{elements}|{bins}|{shape}"


def run_cell(binary, sample, elements, bins, shapes, repeats, min_time, timeout):
    env = dict(os.environ)
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
    shape_axis = "[" + ",".join(shapes) + "]"
    cmd = [
        str(binary), "--benchmark", "base",
        "--axis", f"SampleT{{ct}}={sample}",
        "--axis", f"Elements{{io}}=[{elements}]",
        "--axis", f"Bins=[{bins}]",
        "--axis", f"InputShape={shape_axis}",
        "--min-samples", str(repeats), "--min-time", str(min_time),
        "--timeout", str(timeout), "--csv", "stdout", "--quiet",
    ]
    per_shape = {}
    tag = None
    for _ in range(repeats):
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        tag = tag or ran_tag(p.stderr)
        if p.returncode != 0:
            # OOM / alloc failure at the largest N shows up as a nonzero exit; report it.
            return {}, tag, False, p.stderr
        for r in csv.DictReader(StringIO(p.stdout)):
            if r.get("Skipped") == "Yes":
                continue
            bw = r.get("GlobalMem BW (bytes/sec)", "")
            if bw:
                per_shape.setdefault(r["InputShape"], []).append(float(bw) / 1024**3)
    return {sh: statistics.median(v) for sh, v in per_shape.items() if v}, tag, True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-bin-dir", default="build/autocuda/cub-benchmark/bin")
    ap.add_argument("--out", default="u64_largeN.json")
    ap.add_argument("--bins", type=int, nargs="+", default=DEFAULT_BINS)
    ap.add_argument("--elements", type=int, nargs="+", default=DEFAULT_ELEMENTS)
    ap.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    ap.add_argument("--binaries", nargs="+", default=["even", "range"], choices=list(BINARIES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-time", default="0.02")
    ap.add_argument("--timeout", default="300")
    args = ap.parse_args()

    branch = Path(args.branch_bin_dir)
    results = {}
    calls = 0
    for blabel in args.binaries:
        bbin = branch / BINARIES[blabel]
        if not bbin.exists():
            print(f"!! missing {bbin}; skip {blabel}", flush=True)
            continue
        cells = {}
        selected = cells.setdefault("_selected", {})
        for sample in args.samples:
            for elements in args.elements:
                for bins in args.bins:
                    med, tag, ok, err = run_cell(bbin, sample, elements, bins, args.shapes,
                                                 args.repeats, args.min_time, args.timeout)
                    calls += 1
                    gib = elements * (4 if sample == "I32" else 8) / 1024**3
                    if not ok:
                        why = "OOM/alloc" if ("memory" in err.lower() or "alloc" in err.lower()) else "ABORT"
                        print(f"  {blabel:6} {sample} N={elements:>13} ({gib:.0f}GiB) bins={bins:>8} {why}",
                              flush=True)
                        continue
                    for sh, v in med.items():
                        cells.setdefault("default", {})[cell_key(sample, elements, bins, sh)] = v
                    if tag:
                        for sh in med:
                            selected[cell_key(sample, elements, bins, sh)] = tag
                    line = "  ".join(f"{sh.split(':')[0][:5]}={med[sh]:.0f}" for sh in sorted(med))
                    print(f"  {blabel:6} {sample} N={elements:>13} ({gib:>4.0f}GiB) bins={bins:>8} "
                          f"ran={tag} {line}", flush=True)
        results[blabel] = cells

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({calls} invocations)", flush=True)


if __name__ == "__main__":
    main()
