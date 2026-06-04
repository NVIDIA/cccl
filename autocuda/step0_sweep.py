#!/usr/bin/env python3
"""Step 0 gate: does growing the Direct kernel's SMEM cache past the occupancy-free
point help on skewed inputs? Sweep CUB_HISTO_FORCE_SLOTS at fixed (skewed) cells and
report GiB/s + the slot count the dispatch actually used (from CUB_HISTO_DEBUG_SLOTS).

No kernel changes. Runs the EXISTING single-channel even/range Direct-selected cells.
"""
import csv
import os
import re
import statistics
import subprocess
import sys
from io import StringIO
from pathlib import Path

BIN_DIR = Path("build/autocuda/cub-benchmark/bin")
BINS = {
    "even": "cub.bench.histogram.even.base",
    "range": "cub.bench.histogram.range.base",
}
# Direct-kernel cells: high bins, single channel. powerlaw is the discriminator;
# concentrated:0.0 is the max-contention spike; hash_synonym/stale_resident are
# the cache adversaries. Pair colon-shapes so NVBench doesn't read [a:b] as a range.
SHAPES = "[concentrated:0.0,powerlaw:0.5,hash_synonym,stale_resident]"
ELEMENTS = 67108864            # 64M
BIN_COUNTS = [262144, 1048576] # cuckoo sweet-spot tier + extreme tail
SLOT_FORCES = [0, 4096, 8192, 16384, 32768]  # 0 = auto; rest forced (clamped by dispatch)
REPEATS = 3


def run_cell(binary, bins, force_slots):
    env = dict(os.environ)
    env["CUB_HISTO_DEBUG_SLOTS"] = "1"
    if force_slots:
        env["CUB_HISTO_FORCE_SLOTS"] = str(force_slots)
    else:
        env.pop("CUB_HISTO_FORCE_SLOTS", None)
    cmd = [
        str(BIN_DIR / binary), "--benchmark", "base",
        "--axis", "SampleT{ct}=I32",
        "--axis", f"Elements{{io}}=[{ELEMENTS}]",
        "--axis", f"Bins=[{bins}]",
        "--axis", f"InputShape={SHAPES}",
        "--min-samples", "3", "--min-time", "0.02", "--timeout", "60",
        "--csv", "stdout", "--quiet",
    ]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(f"  ABORT bins={bins} slots={force_slots}: {proc.stderr[-300:]}\n")
        return None, None
    used = None
    m = re.search(r"auto_or_forced_slots=(\d+)", proc.stderr)
    if m:
        used = int(m.group(1))
    out = {}
    for r in csv.DictReader(StringIO(proc.stdout)):
        if r.get("Skipped") == "Yes":
            continue
        bw = r.get("GlobalMem BW (bytes/sec)", "")
        if bw:
            out[r.get("InputShape")] = float(bw) / 1024**3
    return out, used


def main():
    print(f"{'path':5} {'bins':>8} {'shape':16} {'force':>6} {'used':>6} "
          + "  ".join(f"r{i}" for i in range(REPEATS)) + f" {'median':>8}")
    for path, binary in BINS.items():
        for bins in BIN_COUNTS:
            for force in SLOT_FORCES:
                per_shape_runs = {}
                used_slots = None
                ok = True
                for _ in range(REPEATS):
                    out, used = run_cell(binary, bins, force)
                    if out is None:
                        ok = False
                        break
                    used_slots = used
                    for shp, v in out.items():
                        per_shape_runs.setdefault(shp, []).append(v)
                if not ok:
                    continue
                for shp, runs in sorted(per_shape_runs.items()):
                    med = statistics.median(runs)
                    runs_s = "  ".join(f"{v:6.1f}" for v in runs)
                    fr = "auto" if force == 0 else str(force)
                    print(f"{path:5} {bins:>8} {shp:16} {fr:>6} {str(used_slots):>6} {runs_s} {med:8.1f}")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
