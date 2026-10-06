#!/usr/bin/env python3
# Sweep B: larger PRIVATIZED-SMEM cap. For each cap-variant binary
# (cub.bench.histogram.<label>.base.cap<N>, built by build_cap_variants.sh) plus
# the stock 16384-cap baseline binary, measure the bin counts that the larger cap
# brings on-chip. At those bins we compare:
#   - smem_priv_dynamic  (now selected, since num_bins <= cap) -- forced
#   - the high-bin algorithms it would otherwise fall through to (cuckoo /
#     single_probe / gmem_priv_gather / hybrid) -- forced
# so we can see whether the bigger on-chip histogram beats the high-bin path it
# replaces, and how far the win/loss tracks the occupancy drop (cap 16384=64KB=3
# blk/SM, 24576=96KB=2, 32768=128KB=1, 49152=192KB=1 on B200's 228KB/SM).
#
# DO NOT run while another GPU benchmark is active.
import csv
import json
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

# Privatized-cap variant binaries (built by build_cap_variants.sh) + the stock
# baseline. Override with HIST_BENCH_BINDIR. Outputs go to $HIST_SWEEP_OUTDIR
# (default: cwd) so this tracked script never writes results into the source tree.
BINDIR = Path(os.environ.get("HIST_BENCH_BINDIR", "build/autocuda/cub-benchmark/bin"))
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "priv_cap_results.json"

# (cap value, binary suffix). 16384 = stock baseline binary (no suffix).
CAPS = [16384, 24576, 32768, 49152]  # 16384 = shipped baseline (stock binary)
# label -> (stock baseline binary name [dotted], variant base name [underscored,
# as emitted by build_cap_variants.sh]). The cap variants are named with
# underscores (filesystem-safe); the stock binaries keep the dotted nvbench names.
BINARIES = [
    ("even", "cub.bench.histogram.even.base", "cub.bench.histogram.even.base"),
    ("range", "cub.bench.histogram.range.base", "cub.bench.histogram.range.base"),
    (
        "multi_even",
        "cub.bench.histogram.multi.even.base",
        "cub.bench.histogram.multi_even.base",
    ),
    (
        "multi_range",
        "cub.bench.histogram.multi.range.base",
        "cub.bench.histogram.multi_range.base",
    ),
]


def binary_name(stock, variant_base, cap):
    return stock if cap == 16384 else f"{variant_base}.cap{cap}"


# Bins spanning the privatized region the larger caps bring on-chip (24576..49152)
# AND high-bin counts above every cap (65536..1048576), so the high-bin algorithms
# have a full curve to compare priv_dynamic against. For a given cap, only bins <=
# cap route to priv_dynamic (natural selection); the rest fall to the high-bin path.
BINS = [16384, 24576, 32768, 49152, 65536, 131072, 262144, 524288, 1048576]
# Same 9-shape set as the main sweep (was missing temporal_phases + sawtooth).
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
# Full N range: the capacity/occupancy tradeoff is N-dependent (small N = grid
# unsaturated, occupancy ~free; large N = saturated, occupancy costly).
ELEMENTS = [1048576, 16777216, 67108864, 268435456, 1073741824, 2000000000]
# Comparison design (the force hook only fires for bins > the binary's cap, so we
# can't force a high-bin algo on a variant whose cap already covers the bin):
#   * priv_dynamic at a given bin B is measured on the variant whose cap >= B, by
#     NATURAL selection (no force) -- recorded as algo "smem_priv_dynamic".
#   * the high-bin alternatives for the SAME bin B are measured on the BASELINE
#     cap16384 binary (where B > 16384, so forcing works) -- one run per algo.
# So for each cap-variant we run only the natural (unforced) selection; for the
# baseline we additionally force each high-bin algo.
ALGOS_BASELINE_FORCE = [
    "direct_atomic_cuckoo",
    "direct_atomic_single_probe",
    "direct_atomic_no_cache",
    "gmem_priv_gather",
    "hybrid_single_pass",
]

LAUNCH_RE = re.compile(r"\[launch\].*->\s*([a-z_]+)")


def run(binexe, algo):
    cmd = [
        str(binexe),
        "--benchmark",
        "base",
        "--axis",
        "SampleT{ct}=[I32,F64]",
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
    if algo is not None:  # None = natural selection (no force)
        env["CUB_HISTO_FORCE_ALGO"] = algo
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
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
            f"[B] {Path(binexe).name} {algo} exit={p.returncode}\n{p.stderr[-800:]}\n"
        )
        return {}, set()
    launched = set(LAUNCH_RE.findall(p.stderr))
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
    return cells, launched


def main():
    results = {}
    for cap in CAPS:
        for label, stock, variant_base in BINARIES:
            binexe = BINDIR / binary_name(stock, variant_base, cap)
            if not binexe.exists():
                sys.stderr.write(f"[B] MISSING {binexe} (skip; build it first)\n")
                continue
            # Natural (unforced) selection on every cap variant: at bins <= cap this
            # is smem_priv_dynamic; the launch log records what actually ran.
            tag = f"cap{cap}|{label}|natural"
            sys.stderr.write(f"[B] {tag}\n")
            cells, launched = run(binexe, None)
            results[tag] = {
                "cap": cap,
                "label": label,
                "algo": "natural",
                "launched": sorted(launched),
                "cells": cells,
            }
            # On the baseline (cap16384) binary, also force each high-bin algo so we
            # have the alternatives for bins in (16384, 49152] that the larger caps
            # bring on-chip. (Forcing only fires for bins > 16384 here.)
            if cap == 16384:
                for algo in ALGOS_BASELINE_FORCE:
                    tag = f"cap{cap}|{label}|{algo}"
                    sys.stderr.write(f"[B] {tag}\n")
                    cells, launched = run(binexe, algo)
                    results[tag] = {
                        "cap": cap,
                        "label": label,
                        "algo": algo,
                        "launched": sorted(launched),
                        "cells": cells,
                    }
    OUT.write_text(json.dumps(results, indent=1))
    sys.stderr.write(f"[B] wrote {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
