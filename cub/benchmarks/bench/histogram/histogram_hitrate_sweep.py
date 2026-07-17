#!/usr/bin/env python3
# Pass 1 of the main sweep: gather SMEM-cache HIT RATE for the cached high-bin
# algorithms (cuckoo, single_probe). Uses the hit-rate-instrumented binaries
# (built with -DCUB_HISTO_TRACK_HITRATE=1) which, under CUB_HISTO_LOG_HITRATE=1,
# print one '[hitrate] bins=.. ch=.. pixels=.. hits=.. misses=.. rate=..' line per
# cached launch. Measuring hit rate adds overhead, so this is a SEPARATE pass from
# the performance sweep (sweep_algorithms_6shape.py).
#
# Hit rate is a function of the bin-index sequence and the compiled cache policy.
# Although identical bin sequences are sample-type-independent, cache capacity can
# change with SampleT because the selected kernel's occupancy changes. Cache-sensitive
# generators also use that capacity. Measure every plotted SampleT instead of reusing
# I32 measurements for F64. The shape is fixed per subprocess (it is not in the
# [hitrate] line); each line is keyed to its (SampleT, bins, pixels==elements) cell.
#
# Output: hitrate_results.json in autocuda/ (augments the MAIN sweep; the perf
# JSON sweep_results_6shape.json is untouched).
import hashlib
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
QUERY_BINDIR = Path(os.environ.get("HIST_HR_QUERY_BINDIR", str(BINDIR)))
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "hitrate_results.json"

# Same matrix as the main sweep, minus the sample-type axis (hit rate is
# sample-independent: pick one). Cached algos only.
# NOTE: the cap/hitrate variant builder emits filesystem-safe UNDERSCORE names
# (multi_even, not multi.even), unlike the stock dotted nvbench binaries.
# HIST_HR_BINARY_SUFFIX appends a variant suffix (e.g. ".u64") so the 64-bit leg
# targets cub.bench.histogram.<b>.base.hitrate.u64 instead of the 32-bit name.
_HR_SUFFIX = os.environ.get("HIST_HR_BINARY_SUFFIX", "")
BINARIES = [
    (
        "even",
        f"cub.bench.histogram.even.base.hitrate{_HR_SUFFIX}",
        "cub.bench.histogram.even.base",
    ),
    (
        "range",
        f"cub.bench.histogram.range.base.hitrate{_HR_SUFFIX}",
        "cub.bench.histogram.range.base",
    ),
    (
        "multi_even",
        f"cub.bench.histogram.multi_even.base.hitrate{_HR_SUFFIX}",
        "cub.bench.histogram.multi.even.base",
    ),
    (
        "multi_range",
        f"cub.bench.histogram.multi_range.base.hitrate{_HR_SUFFIX}",
        "cub.bench.histogram.multi.range.base",
    ),
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
SAMPLES = _env_list("HIST_HR_SAMPLES", ["I32", "F64"])
# Current CUB_HISTO_FORCE_ALGO names for the two cached high-bin kernels (the old
# direct_atomic_* spellings are no longer recognized -> forcing would silently
# no-op and report no cached launch). These keys are also what histogram_algo_perf.py
# reads for the hit-rate panels.
HITRATE_ALGORITHM_NAMES = {
    "GPS": "gmem_privatized_single_probe",
    "GSC": "gmem_privatized_single_probe_coalesced_spill",
    "GSR": "gmem_privatized_single_probe_rle_spill",
    "DAS": "direct_single_probe",
    "DAC": "direct_cuckoo",
}
ALGOS = [
    HITRATE_ALGORITHM_NAMES.get(name, name)
    for name in os.environ.get("HIST_HR_ALGOS", "DAS DAC").split()
]
GENERATOR_CACHE_SLOTS = int(os.environ.get("HIST_HR_CACHE_SLOTS", "0"))

HITRATE_RE = re.compile(
    r"\[hitrate\] bins=(\d+) ch=(\d+) pixels=(\d+) hits=(\d+) misses=(\d+) rate=([0-9.]+)"
)
INPUT_CACHE_RE = re.compile(r"\[input-cache\] slots=(\d+)")
SLOTS_RE = re.compile(r"\[CUB_HISTO_DEBUG_SLOTS\].*auto_or_forced_slots=(\d+)")
LAUNCH_RE = re.compile(r"\[launch\] bins=(\d+) ch=(\d+) ran=([a-z_:]+)")
_CACHE_SLOT_QUERY_RESULTS = {}
# NVBench may run several warmup/measurement launches per cell; the rate is stable
# across them (deterministic input), so we keep the LAST line seen per (bins,pixels).


def query_cache_slots(binexe, sample, elements):
    """Ask this compiled policy for S without allocating the benchmark input."""
    key = (str(Path(binexe).resolve()), sample, elements)
    if key in _CACHE_SLOT_QUERY_RESULTS:
        return _CACHE_SLOT_QUERY_RESULTS[key]

    cmd = [
        str(binexe),
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


def cache_slot_groups(binexe, sample, elements):
    """Group cells by the cache size queried from the compiled policy."""
    if GENERATOR_CACHE_SLOTS > 0:
        return [(GENERATOR_CACHE_SLOTS, list(elements))]
    groups = {}
    for n in elements:
        groups.setdefault(query_cache_slots(binexe, sample, n), []).append(n)
    return sorted(groups.items())


def run(binexe, algo, sample, shape, elements, stale_slots):
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
        f"SampleT{{ct}}=[{sample}]",
        "--axis",
        "Elements{io}=[" + ",".join(str(e) for e in elements) + "]",
        "--axis",
        "Bins=[" + ",".join(str(b) for b in BINS) + "]",
        "--axis",
        f"InputShape=[{shape},{shape}]",
    ]  # >=2 values: NVBench rejects single-value string axis
    env = dict(os.environ)
    env["CUB_HISTO_FORCE_ALGO"] = algo
    env["CUB_HISTO_FORCE_SLOTS"] = str(stale_slots)
    env["CUB_HISTO_DEBUG_SLOTS"] = "1"
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
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
        raise RuntimeError(
            f"[HR] {Path(binexe).name} {algo} {shape} exit={p.returncode}\n{p.stderr[-600:]}\n"
        )
    launched_slots = {int(match.group(1)) for match in SLOTS_RE.finditer(p.stderr)}
    if launched_slots != {stale_slots}:
        raise RuntimeError(
            f"[HR] {Path(binexe).name} {algo} {sample} {shape}: "
            f"forced slots={stale_slots}, launched={sorted(launched_slots)}"
        )
    launched_algos = {match.group(3) for match in LAUNCH_RE.finditer(p.stderr)}
    if launched_algos != {algo}:
        raise RuntimeError(
            f"[HR] {Path(binexe).name} {algo} {sample} {shape}: "
            f"launch tags={sorted(launched_algos)}"
        )
    cells = {}
    for line in p.stderr.splitlines():
        m = HITRATE_RE.search(line)
        if not m:
            continue
        bins, channels, pixels, hits, misses, rate = m.groups()
        channels = int(channels)
        pixels = int(pixels)
        hits = int(hits)
        misses = int(misses)
        if hits + misses != pixels * channels:
            raise RuntimeError(
                f"[HR] inconsistent contribution count: bins={bins} channels={channels} "
                f"pixels={pixels} hits={hits} misses={misses}"
            )
        cells[f"{int(bins)}|{int(pixels)}"] = {
            "channels": channels,
            "hits": hits,
            "misses": misses,
            "rate": float(rate),
        }
    expected = {f"{bins}|{n}" for bins in BINS for n in elements}
    if cells.keys() != expected:
        raise RuntimeError(
            f"[HR] incomplete output for {Path(binexe).name} {algo} {sample} {shape}: "
            f"missing={sorted(expected - cells.keys())[:5]} "
            f"extra={sorted(cells.keys() - expected)[:5]}"
        )
    return cells


def write_results(results):
    """Durably checkpoint after each completed (binary, algo, sample, shape)."""
    meta = results.setdefault("_meta", {})
    meta["schema"] = "binary -> algorithm -> SampleT|Bins|Elements|InputShape"
    for key, values in (
        ("samples", SAMPLES),
        ("bins", BINS),
        ("elements", ELEMENTS),
        ("shapes", SHAPES),
    ):
        meta[key] = list(dict.fromkeys([*meta.get(key, []), *values]))
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=1) + "\n")
    tmp.replace(OUT)


def main():
    # Merge into any existing results so a re-run for a subset of binaries (e.g.
    # only the multi-channel ones) preserves the already-collected data.
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    only = set(sys.argv[1:])  # optional: labels to run; empty = all
    for label, binname, query_binname in BINARIES:
        if only and label not in only:
            continue
        binexe = BINDIR / binname
        query_binexe = QUERY_BINDIR / query_binname
        if not binexe.is_file():
            raise FileNotFoundError(f"missing hit-rate binary: {binexe}")
        if not query_binexe.is_file():
            raise FileNotFoundError(
                f"missing uninstrumented query binary: {query_binexe}"
            )
        provenance = {
            "hitrate_binary": str(binexe.resolve()),
            "hitrate_sha256": hashlib.sha256(binexe.read_bytes()).hexdigest(),
            "query_binary": str(query_binexe.resolve()),
            "query_sha256": hashlib.sha256(query_binexe.read_bytes()).hexdigest(),
            "variant_suffix": _HR_SUFFIX,
        }
        binary_provenance = results.setdefault("_meta", {}).setdefault(
            "binary_provenance", {}
        )
        variant = _HR_SUFFIX or "default"
        label_provenance = binary_provenance.setdefault(label, {})
        # Backward-compatible upgrade from the former one-record-per-label schema.
        if "hitrate_binary" in label_provenance:
            label_provenance = {"default": label_provenance}
            binary_provenance[label] = label_provenance
        previous = label_provenance.get(variant)
        if previous is not None and previous != provenance:
            raise RuntimeError(
                f"refusing to merge hit-rate data from different binaries for "
                f"{label}/{variant}: "
                f"previous={previous}, current={provenance}"
            )
        label_provenance[variant] = provenance
        binary_results = results.setdefault(label, {})
        for algo in ALGOS:
            algo_results = binary_results.setdefault(algo, {})
            for sample in SAMPLES:
                for shape in SHAPES:
                    expected = {
                        f"{sample}|{bins}|{elements}|{shape}"
                        for bins in BINS
                        for elements in ELEMENTS
                    }
                    if expected.issubset(algo_results):
                        sys.stderr.write(
                            f"[HR] SKIP complete {label} {algo} {sample} {shape}\n"
                        )
                        continue
                    cells = {}
                    for stale_slots, elements in cache_slot_groups(
                        query_binexe, sample, ELEMENTS
                    ):
                        sys.stderr.write(
                            f"[HR] {label} {algo} {sample} {shape} slots={stale_slots} "
                            f"elements={elements}\n"
                        )
                        group_cells = run(
                            binexe, algo, sample, shape, elements, stale_slots
                        )
                        duplicates = cells.keys() & group_cells.keys()
                        if duplicates:
                            raise RuntimeError(
                                f"duplicate hit-rate cells for {label}/{algo}/{sample}/{shape}: "
                                f"{sorted(duplicates)}"
                            )
                        cells.update(group_cells)
                    # Re-key for an unambiguous join with the sample-specific figure.
                    for k, v in cells.items():
                        bins, pixels = k.split("|")
                        algo_results[f"{sample}|{bins}|{pixels}|{shape}"] = v
                    write_results(results)
    write_results(results)
    sys.stderr.write(f"[HR] wrote {OUT}\n")
    for label, av in results.items():
        if label.startswith("_"):
            continue
        for algo, cells in av.items():
            sys.stderr.write(f"[HR] {label} {algo}: {len(cells)} cells\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
