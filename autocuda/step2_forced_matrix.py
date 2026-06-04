#!/usr/bin/env python3
"""Steps 2-3: forced apples-to-apples algorithm matrix.

For each (binary, bins, shape) cell, force EVERY candidate algorithm via
CUB_HISTO_FORCE_ALGO and record GiB/s, correctness (exit 0), and whether a
direct-atomic kernel actually executed (CUB_HISTO_DEBUG_SLOTS tell -> guards
against silent fallback to a different kernel). This is the per-cell comparison
the design doc's promotion decision needs: priv_{cuckoo,single_probe} vs hybrid,
direct_{cuckoo,single_probe}, gmem_priv_gather.
"""
import csv
import os
import statistics
import subprocess
import sys
from io import StringIO
from pathlib import Path

BIN_DIR = Path("build/autocuda/cub-benchmark/bin")
BINARIES = {
    "even": "cub.bench.histogram.even.base",
    "range": "cub.bench.histogram.range.base",
    "multi_even": "cub.bench.histogram.multi.even.base",
    "multi_range": "cub.bench.histogram.multi.range.base",
}
# Algorithms to force. "" = selector default (the incumbent).
ALGOS = ["", "gmem_privatized_nocache", "direct_cuckoo", "direct_single_probe", "gmem_privatized_cuckoo", "gmem_privatized_single_probe"]

# Focused on the proposal's window + the adjacent high-bin tail it was predicted
# to lose. 65536 = hybrid's regime (the target); 262144/1048576 = high-bin tail.
ELEMENTS = "[67108864]"  # 64M: amortizes setup, < 256M gather-tax pivot
BINS = ["65536", "262144", "1048576"]
SHAPES = "[concentrated:0.0,powerlaw:0.5,hash_synonym,stale_resident]"
REPEATS = 3


def run(binary, algo, bins):
    env = dict(os.environ)
    env["CUB_HISTO_DEBUG_SLOTS"] = "1"
    if algo:
        env["CUB_HISTO_FORCE_ALGO"] = algo
    else:
        env.pop("CUB_HISTO_FORCE_ALGO", None)
    cmd = [
        str(BIN_DIR / binary), "--benchmark", "base",
        "--axis", "SampleT{ct}=I32",
        "--axis", f"Elements{{io}}={ELEMENTS}",
        "--axis", f"Bins=[{bins}]",
        "--axis", f"InputShape={SHAPES}",
        "--min-samples", "3", "--min-time", "0.02", "--timeout", "90",
        "--csv", "stdout", "--quiet",
    ]
    p = subprocess.run(cmd, env=env, text=True, capture_output=True)
    direct_ran = "[CUB_HISTO_DEBUG_SLOTS]" in p.stderr
    if p.returncode != 0:
        return None, direct_ran
    out = {}
    for r in csv.DictReader(StringIO(p.stdout)):
        if r.get("Skipped") == "Yes":
            continue
        bw = r.get("GlobalMem BW (bytes/sec)", "")
        if bw:
            out[r["InputShape"]] = float(bw) / 1024**3
    return out, direct_ran


def main():
    out_path = Path("autocuda/results/step2_forced_matrix.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["binary", "bins", "shape", "algo", "gibps_median", "exit_ok", "direct_ran", "runs"])
        for bname, binary in BINARIES.items():
            for bins in BINS:
                for algo in ALGOS:
                    per_shape = {}
                    direct_ran = False
                    ok = True
                    for _ in range(REPEATS):
                        res, dr = run(binary, algo, bins)
                        direct_ran = direct_ran or dr
                        if res is None:
                            ok = False
                            break
                        for shp, v in res.items():
                            per_shape.setdefault(shp, []).append(v)
                    label = algo or "default"
                    if not ok:
                        w.writerow([bname, bins, "*", label, "", 0, int(direct_ran), "ABORT"])
                        print(f"  {bname:11} {bins:>8} {label:20} ABORT/exit!=0", flush=True)
                        continue
                    for shp, runs in sorted(per_shape.items()):
                        med = statistics.median(runs)
                        w.writerow([bname, bins, shp, label, f"{med:.2f}", 1, int(direct_ran),
                                    ";".join(f"{v:.1f}" for v in runs)])
                    line = "  ".join(f"{shp.split(':')[0][:6]}={statistics.median(v):.0f}"
                                     for shp, v in sorted(per_shape.items()))
                    print(f"  {bname:11} {bins:>8} {label:20} dr={int(direct_ran)}  {line}", flush=True)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
