#!/usr/bin/env python3
# Pass 2 of the main sweep: gather SMEM-cache HIT RATE for the cached high-bin
# algorithms (cuckoo, single_probe). Uses the hit-rate-instrumented binaries
# (built with -DCUB_HISTO_TRACK_HITRATE=1) which, under CUB_HISTO_LOG_HITRATE=1,
# print one '[hitrate] bins=.. ch=.. pixels=.. hits=.. misses=.. rate=..' line per
# cached launch. Measuring hit rate adds overhead, so this is a SEPARATE pass from
# the performance sweep (sweep_algorithms_6shape.py).
#
# Hit rate is a pure function of the BIN-INDEX sequence (shape, bins, elements,
# seed) -- independent of SampleT -- so we sweep a single sample type. The shape is
# fixed per subprocess (it is not in the [hitrate] line); each line is keyed to its
# cell by (bins, pixels==elements).
#
# Output: hitrate_results.json in autocuda/ (augments the MAIN sweep; the perf
# JSON sweep_results_6shape.json is untouched).
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Hit-rate-instrumented binaries (built by build_hitrate_variants.sh). Override with
# HIST_BENCH_BINDIR. Outputs go to $HIST_SWEEP_OUTDIR (default: cwd) so this tracked
# script never writes results into the source tree -- run it from autocuda/results/.
BINDIR = Path(os.environ.get("HIST_BENCH_BINDIR",
                             "build/autocuda/cub-benchmark/bin"))
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "hitrate_results.json"

# Same matrix as the main sweep, minus the sample-type axis (hit rate is
# sample-independent: pick one). Cached algos only.
# NOTE: the cap/hitrate variant builder emits filesystem-safe UNDERSCORE names
# (multi_even, not multi.even), unlike the stock dotted nvbench binaries.
BINARIES = [("even", "cub.bench.histogram.even.base.hitrate"),
            ("range", "cub.bench.histogram.range.base.hitrate"),
            ("multi_even", "cub.bench.histogram.multi_even.base.hitrate"),
            ("multi_range", "cub.bench.histogram.multi_range.base.hitrate")]
SHAPES = ["concentrated:1.0", "concentrated:0.25", "concentrated:0.0", "powerlaw:0.5",
          "zipf", "temporal_phases", "sawtooth", "stale_resident", "hash_synonym"]
BINS = [32768, 65536, 131072, 262144, 524288, 1048576]
# Hit rate is essentially N-invariant once N >> bins (the per-block cache reaches
# steady state within the first tiles), so the expensive 1G/2G cells add ~no signal
# while dominating runtime on the instrumented (slower, sync-per-launch) build. Use
# four representative sizes spanning small->large for the "# elements" series.
ELEMENTS = [1048576, 16777216, 67108864, 268435456]
ALGOS = ["direct_atomic_cuckoo", "direct_atomic_single_probe"]
SAMPLE = "I32"  # hit rate is sample-independent

HITRATE_RE = re.compile(r"\[hitrate\] bins=(\d+) ch=(\d+) pixels=(\d+) hits=(\d+) misses=(\d+) rate=([0-9.]+)")
# NVBench may run several warmup/measurement launches per cell; the rate is stable
# across them (deterministic input), so we keep the LAST line seen per (bins,pixels).


def run(binexe, algo, shape):
    # --profile: ONE measured launch per cell (plus a warmup), no NVBench sampling
    # loop. Hit/miss counts are deterministic, so a single launch is exact -- this
    # avoids the per-launch readback-sync overhead exploding across NVBench's normal
    # multi-sample timing loop (which made the full sweep take hours).
    cmd = [str(binexe), "--benchmark", "base", "--profile",
           "--axis", f"SampleT{{ct}}=[{SAMPLE}]",
           "--axis", "Elements{io}=[" + ",".join(str(e) for e in ELEMENTS) + "]",
           "--axis", "Bins=[" + ",".join(str(b) for b in BINS) + "]",
           "--axis", f"InputShape=[{shape},{shape}]"]  # >=2 values: NVBench rejects single-value string axis
    env = dict(os.environ)
    env["CUB_HISTO_FORCE_ALGO"] = algo
    env["CUB_HISTO_LOG_HITRATE"] = "1"
    p = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if p.returncode != 0:
        sys.stderr.write(f"[HR] {Path(binexe).name} {algo} {shape} exit={p.returncode}\n{p.stderr[-600:]}\n")
        return {}
    cells = {}
    for line in p.stderr.splitlines():
        m = HITRATE_RE.search(line)
        if not m:
            continue
        bins, _ch, pixels, hits, misses, rate = m.groups()
        cells[f"{int(bins)}|{int(pixels)}"] = {
            "hits": int(hits), "misses": int(misses), "rate": float(rate)}
    return cells


def main():
    # Merge into any existing results so a re-run for a subset of binaries (e.g.
    # only the multi-channel ones) preserves the already-collected data.
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    only = set(sys.argv[1:])  # optional: labels to run; empty = all
    for label, binname in BINARIES:
        if only and label not in only:
            continue
        binexe = BINDIR / binname
        if not binexe.exists():
            sys.stderr.write(f"[HR] MISSING {binexe} (build hitrate variants first)\n")
            continue
        results[label] = {}
        for algo in ALGOS:
            results[label][algo] = {}
            for shape in SHAPES:
                sys.stderr.write(f"[HR] {label} {algo} {shape}\n")
                cells = run(binexe, algo, shape)
                # re-key to bins|elements|shape for joining with the perf/plot data
                for k, v in cells.items():
                    bins, pixels = k.split("|")
                    results[label][algo][f"{bins}|{pixels}|{shape}"] = v
    OUT.write_text(json.dumps(results, indent=1))
    sys.stderr.write(f"[HR] wrote {OUT}\n")
    for label, av in results.items():
        for algo, cells in av.items():
            sys.stderr.write(f"[HR] {label} {algo}: {len(cells)} cells\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
