#!/usr/bin/env python3
"""Analyze step2_forced_matrix.csv: per cell, rank algorithms and decide whether
priv_{cuckoo,single_probe} is EVER the geomean winner (the promotion criterion).
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

CSV = Path("autocuda/results/step2_forced_matrix.csv")

# (binary,bins,shape,algo) -> gibps
data = {}
for r in csv.DictReader(open(CSV)):
    if r["exit_ok"] != "1" or not r["gibps_median"]:
        continue
    data[(r["binary"], r["bins"], r["shape"], r["algo"])] = float(r["gibps_median"])

binaries = sorted({k[0] for k in data})
algos = ["default", "hybrid", "direct_cuckoo", "direct_single_probe",
         "gmem_priv_gather", "priv_cuckoo", "priv_single_probe"]

# Per-cell winner (exclude 'default' from the contest — it's the incumbent ref —
# and exclude gmem_priv_gather as its forced routing is unreliable).
contestants = ["hybrid", "direct_cuckoo", "direct_single_probe", "priv_cuckoo", "priv_single_probe"]

print("=== Per-(binary,bins,shape) winner among real contestants ===")
priv_wins = 0
priv_best_ratio = 0.0  # best priv/winner ratio seen
cells = sorted({(k[0], k[1], k[2]) for k in data})
win_count = defaultdict(int)
for (b, bins, shape) in cells:
    vals = {a: data.get((b, bins, shape, a)) for a in contestants}
    vals = {a: v for a, v in vals.items() if v is not None}
    if not vals:
        continue
    winner = max(vals, key=vals.get)
    win_count[winner] += 1
    wv = vals[winner]
    priv = max([vals.get("priv_cuckoo", 0), vals.get("priv_single_probe", 0)])
    ratio = priv / wv if wv else 0
    priv_best_ratio = max(priv_best_ratio, ratio)
    if winner.startswith("priv"):
        priv_wins += 1
    flag = "  <-- PRIV WINS" if winner.startswith("priv") else ""
    print(f"  {b:11} {bins:>8} {shape:16} winner={winner:20} {wv:7.0f}  priv_best={priv:7.0f} ({ratio*100:4.0f}%){flag}")

print(f"\n=== Win counts among contestants ===")
for a in contestants:
    print(f"  {a:20} {win_count[a]} cells")
print(f"\npriv-spill wins {priv_wins} / {len(cells)} cells")
print(f"best priv/winner ratio anywhere: {priv_best_ratio*100:.1f}%")

# Geomean per algo per binary (over all bins/shapes where present), for a summary.
print("\n=== Geomean GiB/s per algo per binary (all bins x shapes) ===")
hdr = f"{'binary':12}" + "".join(f"{a[:10]:>12}" for a in algos)
print(hdr)
for b in binaries:
    row = f"{b:12}"
    for a in algos:
        vs = [v for (bb, _, _, aa), v in data.items() if bb == b and aa == a]
        if vs:
            g = math.exp(sum(math.log(x) for x in vs) / len(vs))
            row += f"{g:12.0f}"
        else:
            row += f"{'-':>12}"
    print(row)
