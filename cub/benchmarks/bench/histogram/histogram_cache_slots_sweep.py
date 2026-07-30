#!/usr/bin/env python3
# Sweep A: SMEM cache SIZE (slot count) beyond the shipped occupancy-preserving
# auto-sizer, to test whether spending more SMEM on the per-block cache pays off
# despite the occupancy it costs. Uses the worktree force hooks:
#   CUB_HISTO_FORCE_ALGO  - pin the high-bin algorithm (cuckoo / single_probe)
#   CUB_HISTO_FORCE_SLOTS - pin the per-channel cache slot count (now clamped only
#                           to the device opt-in SMEM, not the occupancy budget)
#   CUB_HISTO_DEBUG_SLOTS - emit the chosen slots + caps
#   CUB_HISTO_LOG_LAUNCH  - emit which kernel + smem actually launched
#
# Records the ACTUAL launched (slots, smem_bytes) per cell and flags any cell that
# fell through to a different kernel (so a clamped/over-cap request is not counted
# as if it ran the requested size). Writes per-cell JSON to this folder.
import csv
import json
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

# Stock benchmark binaries with the force hooks. Override with HIST_BENCH_BINDIR.
# Outputs go to $HIST_SWEEP_OUTDIR (default: cwd) so this tracked script never writes
# results into the source tree -- run it from autocuda/results/.
BINDIR = Path(os.environ.get("HIST_BENCH_BINDIR", "build/autocuda/cub-benchmark/bin"))
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "cache_slots_results.json"

BINARIES = [
    ("even", "cub.bench.histogram.even.base"),
    ("range", "cub.bench.histogram.range.base"),
    ("multi_even", "cub.bench.histogram.multi.even.base"),
    ("multi_range", "cub.bench.histogram.multi.range.base"),
]

# Cache-sensitive shapes only (where the cache can matter): skip uniform/sawtooth.
# Full 9-shape set (matches the main sweep). The cache only matters for shapes with
# bin reuse, but include all so the cache-size lever is evaluated on the same inputs
# as everything else (incl. uniform / temporal_phases / sawtooth).
SHAPES = [
    "concentrated:1.0",
    "concentrated:0.25",
    "concentrated:0.0",
    "powerlaw:0.5",
    "poison",
    "temporal_phases:0.10",
    "sawtooth",
    "stale_resident",
    "hash_synonym",
]
# High-bin region only (the cache is a high-bin feature). 32768 included (forceable).
BINS = [32768, 65536, 131072, 262144, 524288, 1048576]
# Span small-N (grid does not saturate the SMs, so lost occupancy is ~free) to
# large-N (grid saturated, occupancy loss bites) -- the capacity/occupancy
# tradeoff is N-dependent, so the full range is needed to see where big SMEM wins.
ELEMENTS = [1048576, 16777216, 67108864, 268435456, 1073741824, 2000000000]
# The cached algos whose CACHE SIZE we sweep (one line per slot count).
ALGOS = ["direct_atomic_cuckoo", "direct_atomic_single_probe"]
# Slot ladder. "auto" = let the shipped sizer choose (no FORCE_SLOTS). Powers of 2
# only (cache uses mask addressing). Single-ch tops out at 16384 (128KB), multi at
# 2048; over-cap requests are clamped by the dispatch and detected via the log.
SLOT_LADDER = ["auto", 1024, 2048, 4096, 8192, 16384]
# The OTHER high-bin algorithms, shown on every Lever-A figure as reference lines
# (cache slot count is irrelevant to them: no_cache disables the cache, gather and
# hybrid do not use it). Run once each, no slot ladder. This makes Lever A a real
# "is any cache size worth it vs. the alternatives" comparison, matching Lever B.
COMPARE_ALGOS = ["direct_atomic_no_cache", "gmem_priv_gather", "hybrid_single_pass"]

AXES_SAMPLE = "[I32,F64]"
LAUNCH_RE = re.compile(
    r"\[launch\].*->\s*([a-z_]+)(?:\(gated\))?\s*\(smem=(\d+) bytes, slots/ch=(\d+)\)"
)


def run(binname, algo, slots, is_multi):
    binexe = BINDIR / binname
    cmd = [
        str(binexe),
        "--benchmark",
        "base",
        "--axis",
        f"SampleT{{ct}}={AXES_SAMPLE}",
        "--axis",
        "Elements{io}=[" + ",".join(str(e) for e in ELEMENTS) + "]",
        "--axis",
        "Bins=[" + ",".join(str(b) for b in BINS) + "]",
        "--axis",
        "InputShape=[" + ",".join(SHAPES) + "]",
        "--min-samples",
        "3",
        "--min-time",
        "0.01",
        "--timeout",
        "12",
        "--csv",
        "stdout",
        "--quiet",
    ]
    env = dict(os.environ)
    env["CUB_HISTO_FORCE_ALGO"] = algo
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
    if slots != "auto":
        env["CUB_HISTO_FORCE_SLOTS"] = str(slots)
    p = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if p.returncode != 0:
        sys.stderr.write(
            f"[A] {binname} {algo} slots={slots} exit={p.returncode}\n{p.stderr[-800:]}\n"
        )
        return {}, 0, 0, 0
    # Parse launch log: which kernel launched, and the actual smem/slots. The log
    # repeats per launch; take the modal launched-kernel + the max slots seen. The
    # cuckoo/single_probe/no_cache/gather lines carry "(smem=.. slots/ch=..)";
    # hybrid uses a different "(smem_primary=.. gmem_tail=..)" format, so detect the
    # launched-kernel NAME with a permissive fallback regex independent of the tail.
    launched, smem_seen, slots_seen = set(), 0, 0
    for line in p.stderr.splitlines():
        if "[launch]" not in line:
            continue
        nm = re.search(r"->\s*([a-z_]+)", line)
        if nm:
            launched.add(nm.group(1))
        m = LAUNCH_RE.search(line)
        if m:
            smem_seen = max(smem_seen, int(m.group(2)))
            slots_seen = max(slots_seen, int(m.group(3)))
    # Did it actually run the forced algo (else it fell through)?
    ran_forced = (algo in launched) or (
        algo == "direct_atomic_cuckoo" and "direct_atomic_cuckoo" in launched
    )
    cells = {}
    for row in csv.DictReader(StringIO(p.stdout)):
        if row.get("Skipped") == "Yes":
            continue
        raw = row.get("GlobalMem BW (bytes/sec)", "")
        if not raw:
            continue
        try:
            gibs = float(raw) / (1024.0**3)
        except ValueError:
            continue
        if gibs > 0:
            key = "|".join(
                row.get(k, "")
                for k in ("SampleT{ct}", "Elements{io}", "Bins", "InputShape")
            )
            cells[key] = gibs
    return cells, slots_seen, smem_seen, int(ran_forced)


def main():
    results = {}
    for label, binname in BINARIES:
        is_multi = label.startswith("multi")
        results[label] = {}
        for algo in ALGOS:
            for slots in SLOT_LADDER:
                tag = f"{algo}@{slots}"
                sys.stderr.write(f"[A] {label} {tag}\n")
                cells, slots_seen, smem_seen, ran = run(binname, algo, slots, is_multi)
                results[label][tag] = {
                    "requested_slots": slots,
                    "launched_slots": slots_seen,
                    "launched_smem_bytes": smem_seen,
                    "ran_forced": ran,
                    "cells": cells,
                }
        # Reference algorithms: one run each (no cache-size sweep), tagged with the
        # sentinel slot "ref" so the plotter draws them as flat comparison lines.
        for algo in COMPARE_ALGOS:
            tag = f"{algo}@ref"
            sys.stderr.write(f"[A] {label} {tag}\n")
            cells, slots_seen, smem_seen, ran = run(binname, algo, "auto", is_multi)
            results[label][tag] = {
                "requested_slots": "ref",
                "launched_slots": slots_seen,
                "launched_smem_bytes": smem_seen,
                "ran_forced": ran,
                "cells": cells,
            }
    OUT.write_text(json.dumps(results, indent=1))
    # Summary
    sys.stderr.write(f"[A] wrote {OUT}\n")
    for label, av in results.items():
        for tag, rec in av.items():
            sys.stderr.write(
                f"[A] {label} {tag}: launched_slots={rec['launched_slots']} "
                f"smem={rec['launched_smem_bytes']} cells={len(rec['cells'])} ran_forced={rec['ran_forced']}\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
