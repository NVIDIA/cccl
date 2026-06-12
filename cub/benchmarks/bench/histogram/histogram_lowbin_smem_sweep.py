#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Low-bin static-vs-dynamic SMEM comparison sweep.

Question: can the dynamic-SMEM privatized kernel (DeviceHistogramSmemPrivatizedDynamicKernel)
REPLACE the static <=256-bin kernel (DeviceHistogramSmemPrivatizedKernel) so CUB ships one
fewer kernel? At <=256 bins the selector normally always runs the static kernel; the new
CUB_HISTO_FORCE_SMEM={static,dynamic} hook pins which one runs, and the launch tag reports
`ran=smem_privatized:static|:dynamic` so each run is validated to have run the intended kernel.

Three series per (binary, SampleT, Elements, Bins, InputShape) cell:
  - `main`    : upstream-main binary, default dispatch (the baseline)
  - `static`  : this branch, CUB_HISTO_FORCE_SMEM=static
  - `dynamic` : this branch, CUB_HISTO_FORCE_SMEM=dynamic

Output JSON schema matches histogram_algo_perf.py's expectation
({binary: {algo: {cellkey: gibps}}}) so the EXISTING plotter renders it, using algo keys
`main`, `smem_static`, `smem_dynamic`, and a `_selected` map (so the static series is the
plotted `default`). bins default to the <=256 tier (+ a couple of boundary points).
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
    "even": "cub.bench.histogram.even.base",
    "range": "cub.bench.histogram.range.base",
    "multi_even": "cub.bench.histogram.multi.even.base",
    "multi_range": "cub.bench.histogram.multi.range.base",
}

# The user asked for "small bin sizes from zero to 256". Histogram needs >=1 bin; the
# low tier is powers of two up to 256, plus 512 just past the static cap to show the
# crossover where the static kernel is no longer even an option.
DEFAULT_BINS = [16, 32, 64, 128, 256, 512]
DEFAULT_ELEMENTS = [1 << 20, 1 << 24, 1 << 28]  # 1M, 16M, 256M
DEFAULT_SAMPLES = ["I32", "F64"]
DEFAULT_SHAPES = [
    "concentrated:1.0", "concentrated:0.5", "concentrated:0.0",
    "powerlaw:0.5", "zipf:1.0", "hash_synonym", "strided_sweep", "sawtooth",
]

# The static privatized-SMEM kernel is compile-time sized for this many bins
# (cub/.../dispatch_histogram.cuh: max_privatized_smem_bins). Forcing it above this is
# an out-of-bounds access, so the sweep only runs the static series at <= this.
MAX_STATIC_BINS = 256

_LAUNCH_RE = re.compile(r"\[launch\] bins=(\d+) ch=(\d+) ran=([a-z_:]+)")


def ran_tag(stderr: str):
    rans = {m.group(3) for m in _LAUNCH_RE.finditer(stderr)}
    return next(iter(rans)) if len(rans) == 1 else None


def cell_key(sample, elements, bins, shape):
    return f"{sample}|{elements}|{bins}|{shape}"


def run_cell(binary, sample, elements, bins, shapes, repeats, min_time, timeout, force_smem=None):
    env = dict(os.environ)
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
    if force_smem:
        env["CUB_HISTO_FORCE_SMEM"] = force_smem
    else:
        env.pop("CUB_HISTO_FORCE_SMEM", None)
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
            return {}, tag, False
        for r in csv.DictReader(StringIO(p.stdout)):
            if r.get("Skipped") == "Yes":
                continue
            bw = r.get("GlobalMem BW (bytes/sec)", "")
            if bw:
                per_shape.setdefault(r["InputShape"], []).append(float(bw) / 1024**3)
    return {sh: statistics.median(v) for sh, v in per_shape.items() if v}, tag, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-bin-dir", default="build/autocuda/cub-benchmark/bin")
    ap.add_argument("--main-bin-dir", default="")
    ap.add_argument("--out", default="lowbin_smem.json")
    ap.add_argument("--bins", type=int, nargs="+", default=DEFAULT_BINS)
    ap.add_argument("--elements", type=int, nargs="+", default=DEFAULT_ELEMENTS)
    ap.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    ap.add_argument("--binaries", nargs="+", default=["even", "range"], choices=list(BINARIES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-time", default="0.02")
    ap.add_argument("--timeout", default="120")
    args = ap.parse_args()

    branch = Path(args.branch_bin_dir)
    main_dir = Path(args.main_bin_dir) if args.main_bin_dir else None
    # The static kernel only EXISTS at <=256 bins; force_static above 256 falls back to
    # the bin-count rule (dynamic). We validate the tag, so a >256 "static" cell that
    # actually ran dynamic is recorded under `smem_dynamic` only (its true identity).
    EXPECT = {"static": "smem_privatized:static", "dynamic": "smem_privatized:dynamic"}

    results = {}
    calls = 0
    for blabel in args.binaries:
        target = BINARIES[blabel]
        bbin = branch / target
        if not bbin.exists():
            print(f"!! missing {bbin}; skip {blabel}", flush=True)
            continue
        cells = {}
        for sample in args.samples:
            for elements in args.elements:
                for bins in args.bins:
                    for kind, key in (("static", "smem_static"), ("dynamic", "smem_dynamic")):
                        # The STATIC kernel is compile-time sized for <=256 bins; forcing
                        # it above that reads out of bounds (illegal memory access). Skip
                        # static>256 -- the comparison is only meaningful in the static
                        # kernel's actual domain; at 512 we still show dynamic + main to
                        # mark where static is no longer an option.
                        if kind == "static" and bins > MAX_STATIC_BINS:
                            continue
                        med, tag, ok = run_cell(bbin, sample, elements, bins, args.shapes,
                                                args.repeats, args.min_time, args.timeout, force_smem=kind)
                        calls += 1
                        if not ok:
                            print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>5} {key:13} ABORT", flush=True)
                            continue
                        if tag != EXPECT[kind]:
                            print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>5} {key:13} "
                                  f"DROP (ran={tag})", flush=True)
                            continue
                        for sh, v in med.items():
                            cells.setdefault(key, {})[cell_key(sample, elements, bins, sh)] = v
                        line = "  ".join(f"{sh.split(':')[0][:5]}={med[sh]:.0f}" for sh in sorted(med))
                        print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>5} {key:13} {line}", flush=True)
                # main baseline (default dispatch) for this (sample): one column.
                if main_dir and (main_dir / target).exists():
                    for elements in args.elements:
                        for bins in args.bins:
                            med, _t, ok = run_cell(main_dir / target, sample, elements, bins, args.shapes,
                                                   args.repeats, args.min_time, args.timeout)
                            calls += 1
                            if not ok:
                                print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>5} {'main':13} ABORT", flush=True)
                                continue
                            for sh, v in med.items():
                                cells.setdefault("main", {})[cell_key(sample, elements, bins, sh)] = v
                            line = "  ".join(f"{sh.split(':')[0][:5]}={med[sh]:.0f}" for sh in sorted(med))
                            print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>5} {'main':13} {line}", flush=True)
        results[blabel] = cells

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({calls} invocations)", flush=True)


if __name__ == "__main__":
    main()
