#!/usr/bin/env python3
# Per-cell wins table over allalgo_columns.json (one fresh measurement epoch). For each
# (binary, sample, bins, shape) cell, find the cell-max over all algorithm columns present;
# tally each algorithm's strict wins (argmax) and near-best (within 2% of the cell max).
# Coalesce-on and coalesce-off variants are SEPARATE entries with distinct 3-letter tags.
# Reference columns `default`/`main` are excluded from the algorithm contest (default = the
# selector's pick, not its own algorithm; main = a different upstream implementation).
import json
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("allalgo_columns.json")
d = json.load(open(src))
# Drop provenance/meta keys (e.g. _meta) so every remaining top-level key is a binary.
d = {k: v for k, v in d.items() if not k.startswith("_")}

# 3-letter tags. Coalesce-on reuse the established histogram_algo_perf.py tags; the
# *_nocoal variants get distinct tags (same family letter, "0" = coalesce off).
TAG = {
    "smem_static":                  "SST",
    "smem_dynamic":                 "SDY",
    "hybrid":                       "HYB",
    "gmem_privatized_nocache":      "GPN",
    "gmem_privatized_cuckoo":       "GPC",
    "gmem_privatized_single_probe": "GPS",
    "direct_nocache":               "DAN",
    "direct_cuckoo":                "DAC",
    "direct_single_probe":          "DAS",
    # no-coalesce variants (unique tags):
    "gmem_privatized_nocache__nocoal":      "GN0",
    "gmem_privatized_cuckoo__nocoal":       "GC0",
    "gmem_privatized_single_probe__nocoal": "GS0",
    "direct_nocache__nocoal":               "DN0",
    "direct_cuckoo__nocoal":                "DC0",
    "direct_single_probe__nocoal":          "DS0",
}
NAME = {
    "SST": "smem static", "SDY": "smem dynamic", "HYB": "hybrid (smem+gmem)",
    "GPN": "gmem-priv gather", "GPC": "gmem-priv cuckoo", "GPS": "gmem-priv single-probe",
    "DAN": "direct-atomic nocache", "DAC": "direct-atomic cuckoo", "DAS": "direct-atomic single-probe",
    "GN0": "gmem-priv gather + no coalesce", "GC0": "gmem-priv cuckoo + no coalesce",
    "GS0": "gmem-priv single-probe + no coalesce", "DN0": "direct-atomic + no cache + no coalesce",
    "DC0": "direct-atomic + cuckoo + no coalesce", "DS0": "direct-atomic + single-probe + no coalesce",
}
# default = the selector's pick (not its own algorithm); main = a different upstream impl;
# _selected = the per-cell launch-tag map (strings, not GiB/s) the main sweep records.
EXCLUDE = {"default", "main", "_selected"}
NEAR = 0.98  # within 2% of cell best

# Aggregate across all binaries (and also keep per-binary for the breakdown).
agg_present = defaultdict(int)
agg_win = defaultdict(int)
agg_near = defaultdict(int)
per_binary = {}

for binary in d:
    cols = [c for c in d[binary] if c not in EXCLUDE]
    keys = set()
    for c in cols:
        keys.update(d[binary][c])
    pres = defaultdict(int); win = defaultdict(int); near = defaultdict(int)
    ncells = 0
    for k in keys:
        vals = {c: d[binary][c][k] for c in cols if k in d[binary][c] and d[binary][c][k] and d[binary][c][k] > 0}
        if not vals:
            continue
        ncells += 1
        mx = max(vals.values())
        w = max(vals, key=vals.get)
        win[w] += 1; agg_win[w] += 1
        for c, v in vals.items():
            pres[c] += 1; agg_present[c] += 1
            if v >= NEAR * mx:
                near[c] += 1; agg_near[c] += 1
    per_binary[binary] = (ncells, pres, win, near)


def order(cols):
    # stable: by descending aggregate strict wins, then tag
    return sorted(cols, key=lambda c: (-agg_win.get(c, 0), TAG.get(c, c)))


ALL_COLS = [c for c in TAG if any(c in d[b] for b in d)]

print("=" * 92)
print("  HISTOGRAM ALGORITHM WINS — one fresh epoch (this run), 2% near-best band")
print("  coalesce-ON and coalesce-OFF (*0 tags) are separate entries. default/main excluded from contest.")
print("=" * 92)
print(f"\n{'tag':4} {'algorithm':34} {'present':>8} {'wins':>7} {'within2%':>9}   never-best?")
for c in order(ALL_COLS):
    t = TAG[c]; p = agg_present.get(c, 0); w = agg_win.get(c, 0); nb = agg_near.get(c, 0)
    flag = ""
    if p > 0 and w == 0:
        flag = "NEVER strictly best"
    if p > 0 and nb == 0:
        flag = "NEVER within 2% of best"
    print(f"{t:4} {NAME[t]:34} {p:>8} {w:>7} {nb:>9}   {flag}")

print("\n" + "=" * 92)
print("  PER-BINARY strict wins (tag: wins)")
print("=" * 92)
for binary in d:
    ncells, pres, win, near = per_binary[binary]
    parts = [f"{TAG[c]}:{win.get(c,0)}" for c in order(ALL_COLS) if c in d[binary] and pres.get(c, 0) > 0]
    print(f"\n{binary} ({ncells} cells):")
    print("   " + "  ".join(parts))
