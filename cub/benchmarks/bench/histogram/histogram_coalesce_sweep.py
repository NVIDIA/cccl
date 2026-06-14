#!/usr/bin/env python3
# Targeted warp-coalesce sweep: at the non-SMEM-privatized (off-chip) bin tiers and
# the 32-bit-counter inputs, compare the two coalesce-bearing forced algorithms --
# GPN (gmem_privatized_nocache) and DAC (direct_cuckoo) -- WITH vs WITHOUT
# warp-coalescing, against DEF (the selector's default pick) and BAS (upstream main).
#
# Warp-coalesce (`__match_any_sync` -> merge a warp's same-bin lanes into one atomic)
# is a compile-time policy knob. "WITH" = the stock branch binaries; "WITHOUT" = the
# variant built with -DCUB_HISTO_FORCE_WARP_COALESCE=0 (the *.nocoal binaries). The
# question: is coalesce-on the right default at these tiers, or does it regress the
# high-entropy / scattered shapes (stale_resident, sawtooth, strided_sweep, uniform)
# where it merges nothing yet still pays the warp-collective latency?
#
# Output: coalesce_sweep.json  (binary -> series -> "SampleT|Elements|Bins|InputShape" -> GiB/s)
# Series: BAS, DEF, GPN_coal, GPN_nocoal, DAC_coal, DAC_nocoal.
import csv
import json
import os
import statistics
import subprocess
import sys
from io import StringIO
from pathlib import Path

BRANCH_BIN = Path("build/autocuda/cub-benchmark/bin")
MAIN_BIN = Path("/home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin")
OUT = Path(os.environ.get("HIST_SWEEP_OUTDIR", ".")) / "coalesce_sweep.json"

BINARIES = ["even", "range"]  # single-channel: where GPN/DAC/hybrid compete off chip
# Non-SMEM-privatized tiers only (selector leaves SMEM-priv at <=16384 single-channel;
# the off-chip candidates run at >=32768). Include 32768..1M.
BINS = [32768, 65536, 131072, 262144, 1048576]
ELEMENTS = [1 << 24, 1 << 28, 1 << 30]  # 16M, 256M, 1G  (32-bit counter; i32 base binaries)
SAMPLES = ["I32", "F64"]
SHAPES = ["concentrated:1.0", "concentrated:0.75", "concentrated:0.5", "concentrated:0.25",
          "concentrated:0.0", "powerlaw:0.75", "powerlaw:0.5", "powerlaw:0.25", "zipf:1.0",
          "hash_synonym", "stale_resident", "temporal_phases", "strided_sweep", "sawtooth"]

# (series, binary-variant-suffix, force-algo-env, expected launch-tag substring)
# variant "" = stock (coalesce ON); ".nocoal" = -DCUB_HISTO_FORCE_WARP_COALESCE=0.
SERIES = [
    ("BAS",        "MAIN", "",                        None),                      # upstream main, selector default
    ("DEF",        "",     "",                        None),                      # branch selector default (coalesce on)
    ("GPN_coal",   "",       "gmem_privatized_nocache", "gmem_privatized_nocache"),
    ("GPN_nocoal", ".nocoal","gmem_privatized_nocache", "gmem_privatized_nocache"),
    ("DAC_coal",   "",       "direct_cuckoo",           "direct_cuckoo"),
    ("DAC_nocoal", ".nocoal","direct_cuckoo",           "direct_cuckoo"),
]

REPEATS = 3
MIN_TIME = "0.02"
TIMEOUT = "300"


def ran_algo_from_stderr(stderr):
    tag = None
    for line in stderr.splitlines():
        if "[launch]" in line and "ran=" in line:
            tag = line.split("ran=", 1)[1].strip()
    return tag


def run_cell(binary_path, algo_env, sample, elements, bins, shapes):
    """One NVBench call sweeping all shapes; REPEATS times -> {shape: median GiB/s},
    ran_algo, ok, unsupported."""
    env = dict(os.environ)
    env["CUB_HISTO_LOG_LAUNCH"] = "1"
    env.pop("CUB_HISTO_FORCE_ALGO", None)
    if algo_env:
        env["CUB_HISTO_FORCE_ALGO"] = algo_env
    shape_axis = "[" + ",".join(shapes) + "]"
    cmd = [str(binary_path), "--benchmark", "base",
           "--axis", f"SampleT{{ct}}={sample}",
           "--axis", f"Elements{{io}}=[{elements}]",
           "--axis", f"Bins=[{bins}]",
           "--axis", f"InputShape={shape_axis}",
           "--min-samples", str(REPEATS), "--min-time", MIN_TIME,
           "--timeout", TIMEOUT, "--csv", "stdout", "--quiet"]
    per_shape = {}
    ran = None
    for _ in range(REPEATS):
        p = subprocess.run(cmd, env=env, text=True, capture_output=True)
        ran = ran or ran_algo_from_stderr(p.stderr)
        if p.returncode != 0:
            unsupported = "operation not supported" in p.stderr
            return {}, ran, False, unsupported
        for r in csv.DictReader(StringIO(p.stdout)):
            if r.get("Skipped") == "Yes":
                continue
            bw = r.get("GlobalMem BW (bytes/sec)", "")
            if bw:
                per_shape.setdefault(r["InputShape"], []).append(float(bw) / 1024**3)
    return {sh: statistics.median(v) for sh, v in per_shape.items() if v}, ran, True, False


def main():
    results = {b: {s[0]: {} for s in SERIES} for b in BINARIES}
    drops = 0
    total = 0
    for binary in BINARIES:
        for series, variant, algo_env, expect in SERIES:
            bindir = MAIN_BIN if variant == "MAIN" else BRANCH_BIN
            suffix = "" if variant in ("", "MAIN") else variant
            binpath = bindir / f"cub.bench.histogram.{binary}.base{suffix}"
            if not binpath.exists():
                sys.stderr.write(f"MISSING {binpath}\n")
                continue
            for sample in SAMPLES:
                for elements in ELEMENTS:
                    for bins in BINS:
                        med, ran, ok, unsup = run_cell(binpath, algo_env, sample, elements, bins, SHAPES)
                        total += 1
                        tag = f"{binary:5} {series:11} {sample} N={elements:>10} bins={bins:>8}"
                        if not ok:
                            label = "DROP(unsupported)" if (unsup and algo_env) else "ABORT"
                            print(f"  {tag} {label}", flush=True)
                            continue
                        # validate forced launch tag; drop silent fallbacks
                        if expect is not None and ran is not None and expect not in ran:
                            print(f"  {tag} DROP(ran={ran})", flush=True)
                            drops += 1
                            continue
                        key_prefix = f"{sample}|{elements}|{bins}|"
                        for sh, v in med.items():
                            results[binary][series][key_prefix + sh] = v
                        print(f"  {tag} ran={ran} ({len(med)} shapes)", flush=True)
    OUT.write_text(json.dumps(results, indent=0))
    print(f"\nwrote {OUT}  ({total} cells, {drops} forced-tag drops)")


if __name__ == "__main__":
    raise SystemExit(main())
