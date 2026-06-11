#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Exhaustive forced-algorithm histogram sweep, with an upstream-`main` baseline.

For every (binary, SampleT, Elements, Bins, InputShape) cell this forces EACH
high-bin algorithm via `CUB_HISTO_FORCE_ALGO` and records GiB/s, plus the
selector's own pick ("default"). It ALSO runs the same cells on a second set of
binaries built from upstream `main` (which has no force hook) and records main's
default dispatch as a pseudo-algorithm column named `main`. The result is the
apples-to-apples matrix `histogram_algo_perf.py` plots: every candidate algorithm
AND the upstream baseline on one GiB/s-vs-#bins axis per (transform, channels,
SampleT, InputShape).

Forcing is only honored above the high-bin threshold (`max_num_output_bins >
max_dynamic_smem_bins`, i.e. bins > 4096); at/below it every forced algo silently
falls back to `smem_privatized`, so those low-bin cells are recorded once under
`default` only (forcing there is a no-op). `CUB_HISTO_DEBUG_SLOTS` is set so a
direct-atomic kernel that actually ran prints a tell on stderr; we record
`direct_ran` per cell to catch a forced algo that silently fell back.

Output JSON schema (consumed by histogram_algo_perf.py):
  { binary: { algo: { "SampleT|Elements|Bins|InputShape": gibps_median } } }
where `binary` in {even, range, multi_even, multi_range}, and `algo` includes the
six forced names, "default", and "main".

Run (substantial wall time -- scope with the flags):
  python histogram_algo_sweep.py \
      --branch-bin-dir build/autocuda/cub-benchmark/bin \
      --main-bin-dir   ../main-baseline/build/cub-benchmark/bin \
      --out sweep_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from io import StringIO
from pathlib import Path

# NVBench target name per logical binary label.
BINARIES = {
    "even": "cub.bench.histogram.even.base",
    "range": "cub.bench.histogram.range.base",
    "multi_even": "cub.bench.histogram.multi.even.base",
    "multi_range": "cub.bench.histogram.multi.range.base",
}

# Forced high-bin algorithms (env values for CUB_HISTO_FORCE_ALGO). "" == the
# selector's own pick, recorded under the "default" key. `main` is handled
# separately (a different binary, no force hook).
# `hybrid` and `gmem_privatized_nocache` both live under the GmemPrivatized<NoCache>
# kernel (hybrid = smem_split>0 member, gmem_privatized_nocache = pure-gather member);
# forcing pins one member. hybrid is single-channel only and needs a non-empty GMEM
# secondary tail (bins > HYBRID_MIN_BINS); cells outside that domain are SKIPPED up
# front via _forced_algo_applicable (running them would abort, not fall back). The
# post-run DROP check on the [launch] tag remains as a safety net if the floor drifts.
FORCED_ALGOS = [
    "",  # default (selector)
    "hybrid",
    "gmem_privatized_nocache",
    "gmem_privatized_cuckoo",
    "gmem_privatized_single_probe",
    "direct_nocache",
    "direct_cuckoo",
    "direct_single_probe",
]
ALGO_KEY = {"": "default"}  # env value -> JSON key; others map to themselves.

# The exact `[launch] ... ran=X` tag each forced request must produce to count as
# "actually ran the requested algorithm". Both GmemPrivatized<NoCache> members report
# ran=gmem_privatized_nocache, distinguished only by the `:hybrid` suffix -- so the
# two requests validate against different full tags. Everything else maps to itself.
EXPECTED_RAN = {
    "hybrid": "gmem_privatized_nocache:hybrid",
    "gmem_privatized_nocache": "gmem_privatized_nocache",
}

# Smallest bin count at which we sweep the FORCED high-bin algorithms. Below this we
# record ONLY `default` (which the selector runs as smem_privatized across the whole
# low-bin tier). This is a SWEEP-SCOPE policy, not a dispatch limit: forcing is now
# honored at every bin count (the old "forcing is a no-op below the on-chip cap" gate
# was removed -- a forced request that is silently ignored is a bug). But below 32768
# smem_privatized is the only competitive algorithm -- the gather / direct-atomic /
# gmem-privatized kernels lose decisively there -- so force-measuring them at 256..16384
# just burns GPU time for columns that are never selected and never win. We therefore
# start the forced set AT 32768 (the first tier where an off-chip algorithm can matter)
# to keep the sweep tractable. `default` is still recorded at every bin, so the low-bin
# smem_privatized curve is fully covered.
HIGH_BIN_THRESHOLD = 32768

# Per-algorithm STRUCTURAL validity floor (a forced algo that cannot run at a cell
# would make dispatch return cudaErrorNotSupported, which the bench escalates to a
# FATAL temp-size abort + core dump -- not a silent fallback). We therefore SKIP such
# cells outright rather than run-then-abort them, the same way we skip all forced
# algos at bins <= HIGH_BIN_THRESHOLD. Knowing the domain beats discovering it by crash.
#
# `hybrid` is the smem_split>0 member of GmemPrivatized<NoCache>: bins in
# [0, split) accumulate in dyn-SMEM, bins in [split, N) in a per-block GMEM secondary
# tail. Dispatch requires that secondary tail to be non-empty, i.e.
# max_num_output_bins > hybrid_smem_split_bin_single_channel (49152), and the member
# is single-channel only. So forcing hybrid at bins <= 49152 OR on a multi-channel
# binary aborts; we skip those. (For single-channel even/range, max_num_output_bins
# == Bins, so the grid's 32768 cell is below the floor and 65536+ is above it.)
# MUST stay in sync with dispatch_histogram.cuh:hybrid_smem_split_bin_single_channel.
HYBRID_MIN_BINS = 49152


def _forced_algo_applicable(akey: str, bins: int, multichannel: bool) -> bool:
    """Whether forcing `akey` at this (bins, channels) cell can structurally run in
    dispatch. False => dispatch returns cudaErrorNotSupported and the bench aborts, so
    the sweep skips the cell (no run, no abort, no column). All forced algos except
    `hybrid` are valid across the whole high-bin tier; `hybrid` needs a GMEM secondary
    tail (bins > HYBRID_MIN_BINS) and is single-channel only."""
    if akey == "hybrid":
        return (not multichannel) and bins > HYBRID_MIN_BINS
    return True

# Default sweep grid. Bin counts straddle the SMEM tier (<=4096), the gather tier
# (8192..65535), and the direct-atomic tiers (>=65536 cuckoo, >=262144 noprobe2).
DEFAULT_BINS = [256, 2048, 8192, 32768, 65536, 262144, 1048576]
DEFAULT_ELEMENTS = [1 << 24, 1 << 28]  # 16M, 256M
DEFAULT_SAMPLES = ["I32"]
# Full shape set, matching the input-shape characterization figures: the
# concentrated and powerlaw families are swept across their entropy knob (not just
# the endpoints) so the perf grid covers the same inputs the characterization does.
DEFAULT_SHAPES = [
    "concentrated:1.0",
    "concentrated:0.75",
    "concentrated:0.5",
    "concentrated:0.25",
    "concentrated:0.0",
    "powerlaw:0.75",
    "powerlaw:0.5",
    "powerlaw:0.25",
    "zipf:1.0",
    "hash_synonym",
    "stale_resident",
    "temporal_phases",
    "strided_sweep",
    "sawtooth",
]

# InputShape generators comparable against the `main` series. This is now ALL
# shapes the run sweeps, PROVIDED the main-baseline binaries were built with this
# branch's input-shape generators overlaid on stock-main dispatch (i.e. main's
# `histogram_inputs.cuh` / `even.cu` / ... replaced by this branch's, which is a
# bench-only change -- the two rework commits 0cf4594ba6 + 286a78e248 touch no
# dispatch/kernel code). With identical generators every shape is apples-to-apples
# and only the dispatch differs, which is exactly the comparison we want.
#
# Empty set (the default) means "no restriction -- compare every swept shape".
# If you instead point --main-bin-dir at UNMODIFIED upstream-main binaries, set
# this to the generator-identical subset to avoid an apples-to-oranges plot:
#   {"powerlaw:0.5", "zipf:1.0", "hash_synonym", "temporal_phases", "strided_sweep"}
# (the others changed generator on the branch: concentrated:* reweighted,
# concentrated:1.0 reordered, stale_resident reparametrized; sawtooth is branch-only).
MAIN_COMPARABLE_SHAPES: set[str] = set()


def cell_key(sample: str, elements: int, bins: int, shape: str) -> str:
    return f"{sample}|{elements}|{bins}|{shape}"


# Maps the `[launch] ... ran=X` tag the dispatch emits to the canonical algorithm
# name the sweep forces. The hybrid member reports `gmem_privatized_nocache:hybrid`;
# it serves the `gmem_privatized_nocache` enum, so it validates that request.
_LAUNCH_RE = re.compile(r"\[launch\] bins=(\d+) ch=(\d+) ran=([a-z_:]+)")


def _ran_algo_from_stderr(stderr: str):
    """The FULL algorithm tag the dispatch actually launched, parsed from the
    env-gated `[launch]` tag (e.g. `gmem_privatized_nocache` or its `:hybrid` member
    variant). Dispatch routing is shape-blind, so every `[launch]` line in one
    invocation reports the same tag -> return it (None if absent or mixed). The
    `:hybrid` suffix is KEPT so the hybrid vs pure-gather members are distinguishable."""
    rans = {m.group(3) for m in _LAUNCH_RE.finditer(stderr)}
    if len(rans) == 1:
        return next(iter(rans))
    return None  # 0 (no tag) or >1 (mixed -- shouldn't happen for a single-bin invocation)


def run_cell(binary_path, algo_env, sample, elements, bins, shapes, repeats, min_time, timeout):
    """Run one NVBench invocation (all `shapes` in one go) `repeats` times; return
    {shape: median_gibps}, ran_algo, ok_bool. `ran_algo` is the canonical algorithm
    the dispatch ACTUALLY launched (from the CUB_HISTO_LOG_LAUNCH tag), so the caller
    can drop cells where a forced algorithm silently fell back to a different one. A
    single binary call sweeps the InputShape axis, so we pass the whole shape list."""
    env = dict(os.environ)
    env["CUB_HISTO_LOG_LAUNCH"] = "1"  # emit the per-launch "[launch] ... ran=X" tag
    if algo_env:
        env["CUB_HISTO_FORCE_ALGO"] = algo_env
    else:
        env.pop("CUB_HISTO_FORCE_ALGO", None)
    # NVBench rejects range syntax for a single string-axis value, so always pass
    # >= 2 shapes; callers ensure that (we sweep the full shape list at once).
    shape_axis = "[" + ",".join(shapes) + "]"
    cmd = [
        str(binary_path), "--benchmark", "base",
        "--axis", f"SampleT{{ct}}={sample}",
        "--axis", f"Elements{{io}}=[{elements}]",
        "--axis", f"Bins=[{bins}]",
        "--axis", f"InputShape={shape_axis}",
        "--min-samples", str(repeats), "--min-time", str(min_time),
        "--timeout", str(timeout), "--csv", "stdout", "--quiet",
    ]
    per_shape = {}
    ran_algo = None
    for _ in range(repeats):
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        ran_algo = ran_algo or _ran_algo_from_stderr(p.stderr)
        if p.returncode != 0:
            return {}, ran_algo, False
        for r in csv.DictReader(StringIO(p.stdout)):
            if r.get("Skipped") == "Yes":
                continue
            bw = r.get("GlobalMem BW (bytes/sec)", "")
            if bw:
                per_shape.setdefault(r["InputShape"], []).append(float(bw) / 1024**3)
    return {sh: statistics.median(v) for sh, v in per_shape.items() if v}, ran_algo, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch-bin-dir", default="build/autocuda/cub-benchmark/bin",
                    help="dir with this branch's cub.bench.histogram.*.base binaries (force hook present)")
    ap.add_argument("--main-bin-dir", default="",
                    help="dir with upstream main's cub.bench.histogram.*.base binaries (default dispatch only); "
                         "omit to skip the main baseline column")
    ap.add_argument("--out", default="sweep_results.json", help="output per-cell JSON")
    ap.add_argument("--bins", type=int, nargs="+", default=DEFAULT_BINS)
    ap.add_argument("--elements", type=int, nargs="+", default=DEFAULT_ELEMENTS)
    ap.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES, help="SampleT axis values, e.g. I32 F64")
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    ap.add_argument("--binaries", nargs="+", default=list(BINARIES), choices=list(BINARIES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-time", default="0.02")
    ap.add_argument("--timeout", default="120")
    ap.add_argument("--main-comparable-shapes", nargs="*", default=None,
                    help="restrict the `main` baseline to these shapes (use when --main-bin-dir is "
                         "UNMODIFIED upstream main, whose generators differ). Default: all swept shapes "
                         "(correct when the main binaries carry this branch's generators -- see "
                         "MAIN_COMPARABLE_SHAPES).")
    args = ap.parse_args()

    branch_dir = Path(args.branch_bin_dir)
    main_dir = Path(args.main_bin_dir) if args.main_bin_dir else None
    if main_dir and not main_dir.exists():
        raise SystemExit(f"--main-bin-dir does not exist: {main_dir}")

    # main has no force hook, so it contributes ONE column (its default dispatch).
    # Comparable-shape set: CLI override, else MAIN_COMPARABLE_SHAPES, else (empty
    # => no restriction) every swept shape -- correct when the main binaries were
    # built with this branch's input-shape generators (the documented setup).
    if args.main_comparable_shapes is not None:
        comparable = set(args.main_comparable_shapes)
    elif MAIN_COMPARABLE_SHAPES:
        comparable = MAIN_COMPARABLE_SHAPES
    else:
        comparable = set(args.shapes)  # no restriction
    main_shapes = [s for s in args.shapes if s in comparable] if main_dir else []
    if main_dir:
        skipped = [s for s in args.shapes if s not in comparable]
        print(f"main baseline: comparing shapes {sorted(main_shapes)}", flush=True)
        if skipped:
            print(f"  SKIPPING (not in comparable set) {sorted(skipped)}", flush=True)

    results: dict[str, dict[str, dict[str, float]]] = {}
    total_calls = 0
    for blabel in args.binaries:
        target = BINARIES[blabel]
        branch_bin = branch_dir / target
        if not branch_bin.exists():
            print(f"!! missing branch binary {branch_bin}; skipping {blabel}", flush=True)
            continue
        algo_cells: dict[str, dict[str, float]] = {}
        # Ground-truth record of which algorithm the selector `default` actually
        # launched at each cell (from the [launch] tag), so the plotter can label the
        # default series exactly rather than inferring it. Keyed like the perf cells.
        selected = algo_cells.setdefault("_selected", {})
        for sample in args.samples:
            for elements in args.elements:
                for bins in args.bins:
                    high = bins >= HIGH_BIN_THRESHOLD  # forced set starts AT 32768
                    multichannel = blabel.startswith("multi")
                    # Below 32768 record only `default` (smem_privatized is the only
                    # competitive algorithm there -- see HIGH_BIN_THRESHOLD). At/above it,
                    # restrict each forced algo to cells where it can structurally run --
                    # skipping a cell where dispatch would return cudaErrorNotSupported
                    # (which the bench escalates to a FATAL abort, not a fallback).
                    # `default` ("") is always kept.
                    algos = ([a for a in FORCED_ALGOS
                              if a == "" or _forced_algo_applicable(ALGO_KEY.get(a, a), bins, multichannel)]
                             if high else [""])
                    for algo_env in algos:
                        akey = ALGO_KEY.get(algo_env, algo_env)
                        med, ran, ok = run_cell(branch_bin, algo_env, sample, elements, bins,
                                                args.shapes, args.repeats, args.min_time, args.timeout)
                        total_calls += 1
                        if not ok:
                            print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                  f"{akey:28} ABORT", flush=True)
                            continue
                        # DROP a forced cell whose requested algorithm did NOT actually
                        # run -- the dispatch silently substituted a different one (e.g.
                        # forced direct_cuckoo below the threshold ran gather, or a
                        # multi-channel `hybrid` request that is unsupported). Validate
                        # the [launch] tag against the exact tag the request must emit.
                        # `default` is exempt (it IS "whatever the selector picks"); its
                        # actual pick is recorded into `_selected` for the plotter tags.
                        if algo_env:
                            expected = EXPECTED_RAN.get(akey, akey)
                            if ran != expected:
                                print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                      f"{akey:28} DROP (ran={ran})", flush=True)
                                continue
                        else:
                            if ran is not None:
                                for sh in med:
                                    selected[cell_key(sample, elements, bins, sh)] = ran
                        for sh, v in med.items():
                            algo_cells.setdefault(akey, {})[cell_key(sample, elements, bins, sh)] = v
                        line = "  ".join(f"{sh.split(':')[0][:5]}={med[sh]:.0f}" for sh in sorted(med))
                        tagnote = f" ran={ran}" if not algo_env else ""
                        print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                              f"{akey:28}{tagnote} {line}", flush=True)
            # main baseline column for this sample: default dispatch only, computed
            # ONCE per sample over the full elements x bins grid. (This block lives at
            # the `for sample` level, NOT inside `for elements` -- when it was nested in
            # `for elements` it re-ran the whole main grid once per element size, i.e.
            # len(elements)x redundantly, recomputing the slow 1G/2G cells each time and
            # ~doubling total wall time. Data was correct, just wastefully recomputed.)
            if main_dir and main_shapes:
                main_bin = main_dir / target
                if main_bin.exists():
                    for elements in args.elements:
                        for bins in args.bins:
                            med, _ran, ok = run_cell(main_bin, "", sample, elements, bins,
                                                     main_shapes, args.repeats, args.min_time, args.timeout)
                            total_calls += 1
                            if not ok:
                                print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                      f"{'main':28} ABORT", flush=True)
                                continue
                            for sh, v in med.items():
                                algo_cells.setdefault("main", {})[cell_key(sample, elements, bins, sh)] = v
                            line = "  ".join(f"{sh.split(':')[0][:5]}={med[sh]:.0f}" for sh in sorted(med))
                            print(f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                  f"{'main':28}      {line}", flush=True)
                else:
                    print(f"!! missing main binary {main_bin}; no main column for {blabel}", flush=True)
        results[blabel] = algo_cells

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {args.out} ({total_calls} benchmark invocations)", flush=True)


if __name__ == "__main__":
    main()
