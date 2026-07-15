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

Plus a top-level `_meta` key (NOT a binary -- consumers MUST skip keys starting
with "_") recording provenance: the git commit each binary set was built from
(`branch_commit` for the candidate binaries, `main_commit` for the upstream
baseline binaries), plus the sweep axes. This pins every run's numbers to the
exact source they came from. `branch_commit`/`main_commit` are "unknown" if the
binary dir is not inside a git work tree.

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
from io import StringIO
from pathlib import Path

# NVBench target name per logical binary label.
BINARIES = {
    "even": "cub.bench.histogram.even.base",
    "range": "cub.bench.histogram.range.base",
    "multi_even": "cub.bench.histogram.multi.even.base",
    "multi_range": "cub.bench.histogram.multi.range.base",
}


def git_commit_for_path(path) -> str:
    """The full git commit SHA of the work tree that `path` lives in, so a run's
    numbers are pinned to the exact source the benchmarked binaries were built from.
    Returns "unknown" if `path` is not inside a git work tree (or git is absent)."""
    if not path:
        return "unknown"
    # Resolve to absolute so a relative --branch-bin-dir is interpreted against the
    # caller's cwd, not git's; `build/` is typically gitignored but `git -C` still
    # walks up to the enclosing work tree's HEAD.
    try:
        path = Path(path).resolve()
    except OSError:
        return "unknown"
    try:
        sha = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if sha.returncode != 0:
            return "unknown"
        commit = sha.stdout.strip()
        # Note whether that work tree had uncommitted changes when swept.
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            commit += "-dirty"
        return commit or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# Forced high-bin algorithms (env values for CUB_HISTO_FORCE_ALGO). "" == the
# selector's own pick, recorded under the "default" key. `main` is handled
# separately (a different binary, no force hook).
# `hybrid` is the smem_split>0 member of the older GmemPrivatized gather kernel.
# The GPN result key (`gmem_privatized_nocache`) deliberately forces the newer
# cooperative CacheSpillKernel<no_cache_probe, private_block_spill>; the explicit
# force-name mapping below prevents it from silently measuring the older pure-gather
# member again. hybrid is single-channel only and needs a non-empty GMEM secondary
# tail (bins > HYBRID_MIN_BINS); cells outside that domain are SKIPPED up front via
# _forced_algo_applicable (running them would abort, not fall back). The post-run
# DROP check on the [launch] tag remains as a safety net if the floor drifts.
# Forced algorithms come in two regimes:
#   HIGH-BIN (bins >= HIGH_BIN_THRESHOLD): the off-chip candidates that compete in the
#     high-bin region (hybrid / gmem-priv / direct-atomic). Forced via CUB_HISTO_FORCE_ALGO.
#   LOW-BIN  (bins <= MAX_STATIC_BINS): the two smem-privatized kernel instantiations
#     (static vs dynamic), forced via CUB_HISTO_FORCE_SMEM. These let one unified sweep
#     also answer "can the dynamic kernel replace the static <=256 kernel?" -- previously a
#     separate bespoke script. `smem_static` is skipped above MAX_STATIC_BINS (the static
#     kernel is compile-time-sized for 256 bins; forcing it higher reads out of bounds).
# GPS (gmem_privatized_single_probe) is retained as a forced-only characterization
# point. The selector never returns it, but dedicated high-bin runs compare its
# block-private spill + gather against the direct-atomic cache.
#
# WARP-COALESCE VARIANTS: each coalesce-affected off-chip algo is measured BOTH with
# coalescing (the stock binary) and without (the `*__nocoal` key -> the .nocoal binary,
# built -DCUB_HISTO_FORCE_WARP_COALESCE=0). The `__nocoal` suffix is stripped to form the
# CUB_HISTO_FORCE_ALGO value (the kernel is the same; only the policy flag differs, which
# the launch tag does not encode) and selects the .nocoal binary in run_cell. hybrid is
# NOT coalesce-affected (AccumulatePixelsHybrid has no __match_any_sync), so it has no
# __nocoal variant.
_COALESCE_AFFECTED = [
    "gmem_privatized_nocache",
    "direct_nocache",
    "direct_cuckoo",
    "direct_single_probe",
]
_SPILL_POLICY_ALGOS = [
    "gmem_privatized_agent",
    "gmem_privatized_nocache_direct_spill",
    "gmem_privatized_single_probe_coalesced_spill",
    "gmem_privatized_single_probe_rle_spill",
    "gmem_privatized_nocache_rle_spill",
]
FORCED_HIGH_BIN_ALGOS = (
    ["hybrid", "gmem_privatized_single_probe"]
    + _SPILL_POLICY_ALGOS
    + _COALESCE_AFFECTED
    + [a + "__nocoal" for a in _COALESCE_AFFECTED]
)
FORCED_LOW_BIN_ALGOS = ["smem_static", "smem_dynamic"]
FORCED_ALGOS = (
    [""] + FORCED_HIGH_BIN_ALGOS + FORCED_LOW_BIN_ALGOS
)  # "" == default (selector)
ALGO_KEY = {"": "default"}  # env value -> JSON key; others map to themselves.


def _base_algo(akey: str) -> str:
    """Strip the `__nocoal` marker to get the CUB_HISTO_FORCE_ALGO value / launch-tag base."""
    return akey[: -len("__nocoal")] if akey.endswith("__nocoal") else akey


def _force_algo(akey: str) -> str:
    """Map the published series name to the unambiguous dispatch force name."""
    base = _base_algo(akey)
    if base == "gmem_privatized_nocache":
        return "gmem_privatized_nocache_cooperative"
    if base == "gmem_privatized_agent":
        return "gmem_privatized_nocache"
    return base


# The exact `[launch] ... ran=X` tag each forced request must produce to count as
# "actually ran the requested algorithm". Both GmemPrivatized<NoCache> members report
# ran=gmem_privatized_nocache, distinguished only by the `:hybrid` suffix; the two
# smem-privatized members report ran=smem_privatized with a :static / :dynamic suffix.
# Everything else maps to itself.
EXPECTED_RAN = {
    "hybrid": "gmem_privatized_nocache:hybrid",
    "gmem_privatized_nocache": "gmem_privatized_nocache_cooperative",
    "gmem_privatized_agent": "gmem_privatized_nocache",
    "smem_static": "smem_privatized:static",
    "smem_dynamic": "smem_privatized:dynamic",
}

# The static privatized-SMEM kernel is compile-time sized for this many bins
# (dispatch_histogram.cuh: max_privatized_smem_bins). Forcing smem_static above this is an
# out-of-bounds access; the low-bin forced algos only apply at/below it.
MAX_STATIC_BINS = 256

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
    the sweep skips the cell (no run, no abort, no column).
      - smem_static : the static privatized-SMEM kernel, only valid at <= MAX_STATIC_BINS
        (compile-time sized; forcing it higher reads out of bounds).
      - smem_dynamic: the dynamic privatized-SMEM kernel, valid across the low-bin tier.
      - hybrid      : needs a non-empty GMEM secondary tail (bins > HYBRID_MIN_BINS) and
        is single-channel only.
      - everything else (gmem-priv / direct-atomic): valid across the high-bin tier."""
    if akey == "smem_static":
        return bins <= MAX_STATIC_BINS
    if akey == "smem_dynamic":
        return True
    if akey == "hybrid":
        return (not multichannel) and bins > HYBRID_MIN_BINS
    return True


# Which forced-algo env values `algos_for_bin` may emit. None (default) = no restriction
# (historical behaviour). An empty set => `default`-only (selected-vs-baseline). Set from
# --forced-algos in main().
FORCED_ALGO_FILTER = None
INCLUDE_DEFAULT = True


def algos_for_bin(bins: int) -> list:
    """The forced-algo env values to measure at this bin count, optionally including
    `default`. Two regimes, possibly overlapping at neither end of the grid:
      - LOW-BIN  (bins <= MAX_STATIC_BINS): the smem static-vs-dynamic comparison.
      - HIGH-BIN (bins >= HIGH_BIN_THRESHOLD): the off-chip candidates. Between the two
        (e.g. 2048..16384) only `default` is recorded -- smem_privatized is the lone
        competitive algorithm there and the forced columns would be no-ops or losers.
    FORCED_ALGO_FILTER (from --forced-algos) further restricts the forced entries;
    --omit-default independently removes `default` ("")."""
    algos = [""] if INCLUDE_DEFAULT else []
    if bins <= MAX_STATIC_BINS:
        algos += FORCED_LOW_BIN_ALGOS
    # An explicit forced-algorithm request is authoritative even below the
    # historical sweep-scope floor. This matters when the compiled selector's
    # SMEM boundary is lower for a particular family (multi-RANGE is already
    # off-chip at 4096 bins).
    force_high_bin_set = FORCED_ALGO_FILTER is not None and any(
        algo in FORCED_HIGH_BIN_ALGOS for algo in FORCED_ALGO_FILTER
    )
    if bins >= HIGH_BIN_THRESHOLD or force_high_bin_set:
        algos += FORCED_HIGH_BIN_ALGOS
    if FORCED_ALGO_FILTER is not None:
        algos = [a for a in algos if a == "" or a in FORCED_ALGO_FILTER]
    return algos


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
    "powerlaw:0.25",
    "hash_synonym",
    "stale_resident:0.5",
    "stale_resident:0.25",
    "temporal_phases:0.10",
    "strided_sweep",
    "sawtooth",
    "poison",
    "sawtooth:8192:2654435761:1",
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
#   {"powerlaw:0.25", "strided_sweep"}
# (the others changed generator on the branch: concentrated:* reweighted,
# concentrated:1.0 reordered, stale_resident reparametrized; sawtooth is branch-only).
MAIN_COMPARABLE_SHAPES: set[str] = set()


def cell_key(sample: str, elements: int, bins: int, shape: str) -> str:
    return f"{sample}|{elements}|{bins}|{shape}"


_INPUT_CACHE_RE = re.compile(r"\[input-cache\] slots=(\d+)")
_CACHE_SLOT_QUERY_RESULTS: dict[tuple[str, str, int], int] = {}


def query_cache_slots_for_cell(
    branch_bin: Path, sample: str, elements: int, timeout: str
) -> int:
    """Ask the compiled branch policy for S without allocating the input.

    The returned value is then supplied to both branch and overlaid-main
    generators, guaranteeing byte-identical cache-sensitive inputs without a
    duplicated Python model of CUB's occupancy policy.
    """
    key = (str(branch_bin.resolve()), sample, elements)
    if key in _CACHE_SLOT_QUERY_RESULTS:
        return _CACHE_SLOT_QUERY_RESULTS[key]

    cmd = [
        str(branch_bin),
        "--benchmark",
        "base",
        "--profile",
        "--quiet",
        "--axis",
        f"SampleT{{ct}}={sample}",
        "--axis",
        f"Elements{{io}}={elements}",
        "--axis",
        "Bins=32",
        "--axis",
        "InputShape=[concentrated:0.0,concentrated:0.0]",
    ]
    env = dict(os.environ)
    env.pop("CUB_HISTO_INPUT_CACHE_SLOTS", None)
    env.pop("CUB_HISTO_STALE_SLOTS", None)
    env["CUB_HISTO_LOG_INPUT_CACHE_SLOTS"] = "1"
    env["CUB_HISTO_QUERY_INPUT_CACHE_SLOTS_ONLY"] = "1"
    env["CUB_BENCH_HISTOGRAM_VERIFY"] = "0"
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=max(30, int(float(timeout))),
    )
    slots = {int(match.group(1)) for match in _INPUT_CACHE_RE.finditer(result.stderr)}
    if result.returncode != 0 or len(slots) != 1 or next(iter(slots), 0) <= 0:
        raise RuntimeError(
            f"cache-slot query failed for {branch_bin.name} {sample} N={elements}: "
            f"exit={result.returncode}, slots={sorted(slots)}, stderr={result.stderr[-600:]}"
        )
    value = slots.pop()
    _CACHE_SLOT_QUERY_RESULTS[key] = value
    print(
        f"  queried input cache: {branch_bin.name} {sample} N={elements} -> S={value}",
        flush=True,
    )
    return value


# Parses the exact `[launch] ... ran=X` tag emitted by dispatch. In particular,
# hybrid reports the older gather member while GPN must report the distinct new
# cooperative no-cache/private-spill specialization.
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
    return (
        None  # 0 (no tag) or >1 (mixed -- shouldn't happen for a single-bin invocation)
    )


def run_cell(
    binary_path,
    algo_env,
    sample,
    elements,
    bins,
    cache_slots,
    shapes,
    repeats,
    min_time,
    timeout,
):
    """Run one NVBench invocation (all `shapes` in one go) `repeats` times; return
    {shape: median_gibps}, ran_algo, ok_bool. `ran_algo` is the canonical algorithm
    the dispatch ACTUALLY launched (from the CUB_HISTO_LOG_LAUNCH tag), so the caller
    can drop cells where a forced algorithm silently fell back to a different one. A
    single binary call sweeps the InputShape axis, so we pass the whole shape list."""
    env = dict(os.environ)
    env["CUB_HISTO_LOG_LAUNCH"] = "1"  # emit the per-launch "[launch] ... ran=X" tag
    env["CUB_HISTO_INPUT_CACHE_SLOTS"] = str(cache_slots)
    env.pop("CUB_HISTO_FORCE_ALGO", None)
    env.pop("CUB_HISTO_FORCE_SMEM", None)
    if algo_env in ("smem_static", "smem_dynamic"):
        # Low-bin static-vs-dynamic privatized-SMEM comparison: routed via FORCE_SMEM.
        env["CUB_HISTO_FORCE_SMEM"] = algo_env[len("smem_") :]  # "static" / "dynamic"
    elif algo_env:
        env["CUB_HISTO_FORCE_ALGO"] = algo_env
    # NVBench rejects range syntax for a single string-axis value, so always pass
    # >= 2 shapes; callers ensure that (we sweep the full shape list at once).
    shape_axis = "[" + ",".join(shapes) + "]"
    cmd = [
        str(binary_path),
        "--benchmark",
        "base",
        "--axis",
        f"SampleT{{ct}}={sample}",
        "--axis",
        f"Elements{{io}}=[{elements}]",
        "--axis",
        f"Bins=[{bins}]",
        "--axis",
        f"InputShape={shape_axis}",
        "--min-samples",
        str(repeats),
        "--min-time",
        str(min_time),
        "--timeout",
        str(timeout),
        "--csv",
        "stdout",
        "--quiet",
    ]
    per_shape = {}
    ran_algo = None
    for _ in range(repeats):
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        ran_algo = ran_algo or _ran_algo_from_stderr(p.stderr)
        if p.returncode != 0:
            # Distinguish a STRUCTURAL "this algo can't run on this cell" from a real
            # abort. The dispatch returns cudaErrorNotSupported when a forced algorithm
            # cannot be launched here (e.g. forced pure-gather with a wide counter at a
            # high bin count: its per-block GMEM slabs need the full co-resident grid,
            # which an 8-byte counter's lower occupancy cannot provide), and the bench
            # surfaces that as "FATAL ... -> operation not supported". That is the forced
            # request being correctly DECLINED, not a crash -- the caller drops the cell
            # rather than flagging an abort.
            unsupported = "operation not supported" in p.stderr
            return {}, ran_algo, False, unsupported
        for r in csv.DictReader(StringIO(p.stdout)):
            if r.get("Skipped") == "Yes":
                continue
            bw = r.get("GlobalMem BW (bytes/sec)", "")
            if bw:
                per_shape.setdefault(r["InputShape"], []).append(float(bw) / 1024**3)
    return (
        {sh: statistics.median(v) for sh, v in per_shape.items() if v},
        ran_algo,
        True,
        False,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--branch-bin-dir",
        default="build/autocuda/cub-benchmark/bin",
        help="dir with this branch's cub.bench.histogram.*.base binaries (force hook present)",
    )
    ap.add_argument(
        "--main-bin-dir",
        default="",
        help="dir with upstream main's cub.bench.histogram.*.base binaries (default dispatch only); "
        "omit to skip the main baseline column",
    )
    ap.add_argument("--out", default="sweep_results.json", help="output per-cell JSON")
    ap.add_argument("--bins", type=int, nargs="+", default=DEFAULT_BINS)
    ap.add_argument("--elements", type=int, nargs="+", default=DEFAULT_ELEMENTS)
    ap.add_argument(
        "--samples",
        nargs="+",
        default=DEFAULT_SAMPLES,
        help="SampleT axis values, e.g. I32 F64",
    )
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    ap.add_argument(
        "--binaries", nargs="+", default=list(BINARIES), choices=list(BINARIES)
    )
    ap.add_argument(
        "--binary-suffix",
        default="",
        help="appended to the bench target name for BOTH branch and main bins "
        "(default '' -> cub.bench.histogram.<b>.base; e.g. '.u64' selects the "
        "u32-local/u64-global counter variant cub.bench.histogram.<b>.base.u64). One unified "
        "driver covers the I32, low-bin static/dynamic, and wide-output sweeps.",
    )
    ap.add_argument(
        "--generator-cache-slots",
        type=int,
        default=0,
        help="override the cache-slot count supplied to cache-sensitive input generators "
        "for every cell in this invocation; required for build variants whose LocalCounter "
        "width differs from the default 32-bit dispatch assumptions",
    )
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-time", default="0.02")
    ap.add_argument("--timeout", default="120")
    ap.add_argument(
        "--main-comparable-shapes",
        nargs="*",
        default=None,
        help="restrict the `main` baseline to these shapes (use when --main-bin-dir is "
        "UNMODIFIED upstream main, whose generators differ). Default: all swept shapes "
        "(correct when the main binaries carry this branch's generators -- see "
        "MAIN_COMPARABLE_SHAPES).",
    )
    ap.add_argument(
        "--forced-algos",
        nargs="+",
        default=None,
        metavar="ALGO",
        help="forced-algo columns besides `default` (selector) and the optional `main` baseline. "
        "'all' (default) = every regime-appropriate forced "
        "algo; 'none' = selected-vs-baseline only; or an explicit list.",
    )
    ap.add_argument(
        "--omit-default",
        action="store_true",
        help="do not benchmark the shipping selector; useful for forced-candidate-only studies",
    )
    args = ap.parse_args()

    global FORCED_ALGO_FILTER, INCLUDE_DEFAULT
    INCLUDE_DEFAULT = not args.omit_default
    if args.forced_algos is None or args.forced_algos == ["all"]:
        FORCED_ALGO_FILTER = None
    elif args.forced_algos == ["none"]:
        FORCED_ALGO_FILTER = set()
    else:
        valid = set(FORCED_HIGH_BIN_ALGOS + FORCED_LOW_BIN_ALGOS)
        bad = [a for a in args.forced_algos if a not in valid]
        if bad:
            raise SystemExit(
                f"--forced-algos: unknown {bad}; choose from {sorted(valid)} or all/none"
            )
        FORCED_ALGO_FILTER = set(args.forced_algos)

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
        target = (
            BINARIES[blabel] + args.binary_suffix
        )  # "" (default .base) or e.g. ".u64"
        branch_bin = branch_dir / target
        if not branch_bin.exists():
            print(
                f"!! missing branch binary {branch_bin}; skipping {blabel}", flush=True
            )
            continue
        algo_cells: dict[str, dict[str, float]] = {}
        # Ground-truth record of which algorithm the selector `default` actually
        # launched at each cell (from the [launch] tag), so the plotter can label the
        # default series exactly rather than inferring it. Keyed like the perf cells.
        selected = algo_cells.setdefault("_selected", {})
        for sample in args.samples:
            for elements in args.elements:
                cache_slots = args.generator_cache_slots or query_cache_slots_for_cell(
                    branch_bin, sample, elements, args.timeout
                )
                for bins in args.bins:
                    multichannel = blabel.startswith("multi")
                    # Forced algos for this bin (low-bin smem static/dynamic and/or
                    # high-bin off-chip candidates), each further restricted to cells
                    # where it can structurally run -- skipping a cell where dispatch
                    # would return cudaErrorNotSupported (which the bench escalates to a
                    # FATAL abort, not a fallback). `default` ("") is kept unless
                    # this forced-candidate study requested --omit-default.
                    algos = [
                        a
                        for a in algos_for_bin(bins)
                        if a == ""
                        or _forced_algo_applicable(
                            ALGO_KEY.get(a, a), bins, multichannel
                        )
                    ]
                    for algo_env in algos:
                        akey = ALGO_KEY.get(algo_env, algo_env)
                        # `*__nocoal` keys force their BASE algo (the kernel is identical;
                        # only the warp-coalesce policy flag differs) but run against the
                        # .nocoal binary (built -DCUB_HISTO_FORCE_WARP_COALESCE=0). All other
                        # keys run against this binary with their own name as the force value.
                        force_env = _force_algo(akey) if akey else ""
                        # The .nocoal variant must carry the SAME binary-suffix as the
                        # stock binary (e.g. ".u64"), so the 64-bit leg finds
                        # `<bin>.base.nocoal.u64`, not the 32-bit `<bin>.base.nocoal`.
                        cell_bin = (
                            branch_dir
                            / (BINARIES[blabel] + ".nocoal" + args.binary_suffix)
                            if akey.endswith("__nocoal")
                            else branch_bin
                        )
                        if akey.endswith("__nocoal") and not cell_bin.exists():
                            print(
                                f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                f"{akey:28} SKIP (no {cell_bin.name})",
                                flush=True,
                            )
                            continue
                        med, ran, ok, unsupported = run_cell(
                            cell_bin,
                            force_env,
                            sample,
                            elements,
                            bins,
                            cache_slots,
                            args.shapes,
                            args.repeats,
                            args.min_time,
                            args.timeout,
                        )
                        total_calls += 1
                        if not ok:
                            # A FORCED algo that returns "operation not supported" is being
                            # structurally declined (it cannot launch on this cell -- e.g.
                            # forced pure-gather with a wide counter at a high bin count),
                            # NOT crashing: record it as a DROP, like a launch-tag mismatch.
                            # `default` (algo_env == "") must always run, so an unsupported
                            # there is a real ABORT. A non-unsupported failure is an ABORT.
                            label = (
                                "DROP (unsupported)"
                                if (unsupported and algo_env)
                                else "ABORT"
                            )
                            print(
                                f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                f"{akey:28} {label}",
                                flush=True,
                            )
                            continue
                        # DROP a forced cell whose requested algorithm did NOT actually
                        # run -- the dispatch silently substituted a different one (e.g.
                        # forced direct_cuckoo below the threshold ran gather, or a
                        # multi-channel `hybrid` request that is unsupported). Validate
                        # the [launch] tag against the exact tag the request must emit.
                        # `default` is exempt (it IS "whatever the selector picks"); its
                        # actual pick is recorded into `_selected` for the plotter tags.
                        if algo_env:
                            # `*__nocoal` emits the SAME launch tag as its base algo
                            # (coalesce on/off is not encoded in the tag), so validate
                            # against the base's expected tag.
                            base = _base_algo(akey)
                            expected = EXPECTED_RAN.get(base, base)
                            if ran != expected:
                                print(
                                    f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                    f"{akey:28} DROP (ran={ran})",
                                    flush=True,
                                )
                                continue
                        else:
                            if ran is not None:
                                for sh in med:
                                    selected[cell_key(sample, elements, bins, sh)] = ran
                        for sh, v in med.items():
                            algo_cells.setdefault(akey, {})[
                                cell_key(sample, elements, bins, sh)
                            ] = v
                        line = "  ".join(
                            f"{sh.split(':')[0][:5]}={med[sh]:.0f}"
                            for sh in sorted(med)
                        )
                        tagnote = f" ran={ran}" if not algo_env else ""
                        print(
                            f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                            f"{akey:28}{tagnote} {line}",
                            flush=True,
                        )
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
                        cache_slots = (
                            args.generator_cache_slots
                            or query_cache_slots_for_cell(
                                branch_bin, sample, elements, args.timeout
                            )
                        )
                        for bins in args.bins:
                            med, _ran, ok, _unsup = run_cell(
                                main_bin,
                                "",
                                sample,
                                elements,
                                bins,
                                cache_slots,
                                main_shapes,
                                args.repeats,
                                args.min_time,
                                args.timeout,
                            )
                            total_calls += 1
                            if not ok:
                                print(
                                    f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                    f"{'main':28} ABORT",
                                    flush=True,
                                )
                                continue
                            for sh, v in med.items():
                                algo_cells.setdefault("main", {})[
                                    cell_key(sample, elements, bins, sh)
                                ] = v
                            line = "  ".join(
                                f"{sh.split(':')[0][:5]}={med[sh]:.0f}"
                                for sh in sorted(med)
                            )
                            print(
                                f"  {blabel:11} {sample} N={elements:>10} bins={bins:>8} "
                                f"{'main':28}      {line}",
                                flush=True,
                            )
                else:
                    print(
                        f"!! missing main binary {main_bin}; no main column for {blabel}",
                        flush=True,
                    )
        results[blabel] = algo_cells

    # Provenance: pin this run's numbers to the exact source the binaries were built
    # from. Top-level `_meta` key (consumers skip "_"-prefixed keys -- not a binary).
    results["_meta"] = {
        "branch_commit": git_commit_for_path(args.branch_bin_dir),
        "main_commit": git_commit_for_path(args.main_bin_dir)
        if args.main_bin_dir
        else None,
        "branch_bin_dir": args.branch_bin_dir,
        "main_bin_dir": args.main_bin_dir or None,
        "binary_suffix": args.binary_suffix or None,
        "generator_cache_slots": args.generator_cache_slots or None,
        "binaries": list(args.binaries),
        "samples": list(args.samples),
        "bins": list(args.bins),
        "elements": list(args.elements),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    bc = results["_meta"]["branch_commit"]
    print(
        f"\nwrote {args.out} ({total_calls} benchmark invocations; branch_commit={bc})",
        flush=True,
    )


if __name__ == "__main__":
    main()
