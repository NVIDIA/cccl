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
BINDIR = Path(os.environ.get("HIST_BENCH_BINDIR", "build/autocuda/cub-benchmark/bin"))
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "hitrate_results.json"

# Same matrix as the main sweep, minus the sample-type axis (hit rate is
# sample-independent: pick one). Cached algos only.
# NOTE: the cap/hitrate variant builder emits filesystem-safe UNDERSCORE names
# (multi_even, not multi.even), unlike the stock dotted nvbench binaries.
# HIST_HR_BINARY_SUFFIX appends a variant suffix (e.g. ".u64") so the 64-bit leg
# targets cub.bench.histogram.<b>.base.hitrate.u64 instead of the 32-bit name.
_HR_SUFFIX = os.environ.get("HIST_HR_BINARY_SUFFIX", "")
BINARIES = [
    ("even", f"cub.bench.histogram.even.base.hitrate{_HR_SUFFIX}"),
    ("range", f"cub.bench.histogram.range.base.hitrate{_HR_SUFFIX}"),
    ("multi_even", f"cub.bench.histogram.multi_even.base.hitrate{_HR_SUFFIX}"),
    ("multi_range", f"cub.bench.histogram.multi_range.base.hitrate{_HR_SUFFIX}"),
]


# Match the main perf sweep's full shape set (concentrated/powerlaw swept across
# their entropy knob, not just endpoints). Each axis is overridable via an env var
# (space-separated) so a quick smoke run can shrink the grid without editing.
def _env_list(name, default, cast=str):
    v = os.environ.get(name)
    return [cast(x) for x in v.split()] if v else default


SHAPES = _env_list(
    "HIST_HR_SHAPES",
    [
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
    ],
)
BINS = _env_list("HIST_HR_BINS", [32768, 65536, 131072, 262144, 524288, 1048576], int)
# All input sizes (matching the perf sweep's element axis), so the hit-rate panels
# carry one series per element count over the full small->large range. (Hit rate is
# largely N-invariant once N >> bins, but we sweep every size so nothing is dropped.)
ELEMENTS = _env_list(
    "HIST_HR_ELEMENTS",
    [1048576, 16777216, 67108864, 268435456, 1073741824, 2000000000],
    int,
)
# Current CUB_HISTO_FORCE_ALGO names for the two cached high-bin kernels (the old
# direct_atomic_* spellings are no longer recognized -> forcing would silently
# no-op and report no cached launch). These keys are also what histogram_algo_perf.py
# reads for the hit-rate panels.
ALGOS = ["direct_cuckoo", "direct_single_probe"]
SAMPLE = "I32"  # hit rate is sample-independent
GENERATOR_CACHE_SLOTS = int(os.environ.get("HIST_HR_CACHE_SLOTS", "0"))

HITRATE_RE = re.compile(
    r"\[hitrate\] bins=(\d+) ch=(\d+) pixels=(\d+) hits=(\d+) misses=(\d+) rate=([0-9.]+)"
)
INPUT_CACHE_RE = re.compile(r"\[input-cache\] slots=(\d+)")
_CACHE_SLOT_QUERY_RESULTS = {}
# NVBench may run several warmup/measurement launches per cell; the rate is stable
# across them (deterministic input), so we keep the LAST line seen per (bins,pixels).


def query_cache_slots(binexe, elements):
    """Ask this compiled policy for S without allocating the benchmark input."""
    key = (str(Path(binexe).resolve()), elements)
    if key in _CACHE_SLOT_QUERY_RESULTS:
        return _CACHE_SLOT_QUERY_RESULTS[key]

    cmd = [
        str(binexe),
        "--benchmark",
        "base",
        "--profile",
        "--quiet",
        "--axis",
        f"SampleT{{ct}}={SAMPLE}",
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
    )
    slots = {int(match.group(1)) for match in INPUT_CACHE_RE.finditer(result.stderr)}
    if result.returncode != 0 or len(slots) != 1 or next(iter(slots), 0) <= 0:
        raise RuntimeError(
            f"cache-slot query failed for {Path(binexe).name} N={elements}: "
            f"exit={result.returncode}, slots={sorted(slots)}, stderr={result.stderr[-600:]}"
        )
    value = slots.pop()
    _CACHE_SLOT_QUERY_RESULTS[key] = value
    return value


def cache_slot_groups(binexe, elements):
    """Group cells by the cache size queried from the compiled policy."""
    if GENERATOR_CACHE_SLOTS > 0:
        return [(GENERATOR_CACHE_SLOTS, list(elements))]
    groups = {}
    for n in elements:
        groups.setdefault(query_cache_slots(binexe, n), []).append(n)
    return sorted(groups.items())


def run(binexe, algo, shape, elements, stale_slots):
    # --profile: ONE measured launch per cell (plus a warmup), no NVBench sampling
    # loop. Hit/miss counts are deterministic, so a single launch is exact -- this
    # avoids the per-launch readback-sync overhead exploding across NVBench's normal
    # multi-sample timing loop (which made the full sweep take hours).
    cmd = [
        str(binexe),
        "--benchmark",
        "base",
        "--profile",
        "--axis",
        f"SampleT{{ct}}=[{SAMPLE}]",
        "--axis",
        "Elements{io}=[" + ",".join(str(e) for e in elements) + "]",
        "--axis",
        "Bins=[" + ",".join(str(b) for b in BINS) + "]",
        "--axis",
        f"InputShape=[{shape},{shape}]",
    ]  # >=2 values: NVBench rejects single-value string axis
    env = dict(os.environ)
    env["CUB_HISTO_FORCE_ALGO"] = algo
    env["CUB_HISTO_LOG_HITRATE"] = "1"
    # Always pin the exact per-cell group size. Ambient shell state must not
    # override this or branch/main input sequences cease to be comparable.
    env["CUB_HISTO_INPUT_CACHE_SLOTS"] = str(stale_slots)
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
            f"[HR] {Path(binexe).name} {algo} {shape} exit={p.returncode}\n{p.stderr[-600:]}\n"
        )
        return {}
    cells = {}
    for line in p.stderr.splitlines():
        m = HITRATE_RE.search(line)
        if not m:
            continue
        bins, _ch, pixels, hits, misses, rate = m.groups()
        cells[f"{int(bins)}|{int(pixels)}"] = {
            "hits": int(hits),
            "misses": int(misses),
            "rate": float(rate),
        }
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
        binary_results = results.setdefault(label, {})
        for algo in ALGOS:
            algo_results = binary_results.setdefault(algo, {})
            for shape in SHAPES:
                cells = {}
                for stale_slots, elements in cache_slot_groups(binexe, ELEMENTS):
                    sys.stderr.write(
                        f"[HR] {label} {algo} {shape} slots={stale_slots} "
                        f"elements={elements}\n"
                    )
                    group_cells = run(binexe, algo, shape, elements, stale_slots)
                    duplicates = cells.keys() & group_cells.keys()
                    if duplicates:
                        raise RuntimeError(
                            f"duplicate hit-rate cells for {label}/{algo}/{shape}: "
                            f"{sorted(duplicates)}"
                        )
                    cells.update(group_cells)
                # re-key to bins|elements|shape for joining with the perf/plot data
                for k, v in cells.items():
                    bins, pixels = k.split("|")
                    algo_results[f"{bins}|{pixels}|{shape}"] = v
    OUT.write_text(json.dumps(results, indent=1))
    sys.stderr.write(f"[HR] wrote {OUT}\n")
    for label, av in results.items():
        for algo, cells in av.items():
            sys.stderr.write(f"[HR] {label} {algo}: {len(cells)} cells\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
