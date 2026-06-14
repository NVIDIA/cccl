#!/usr/bin/env python3
# Analyze coalesce_sweep.json: at each off-chip bin tier, does warp-coalescing help or
# hurt GPN and DAC, split by input entropy? Reports per-shape coal/nocoal ratios and the
# headline "what would an input-adaptive coalesce choice buy over the always-on default".
import json
import math
import sys
from pathlib import Path

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("coalesce_sweep.json")
d = json.load(open(src))


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# Group shapes by entropy class for the headline.
LOW_ENTROPY = {"concentrated:0.0", "concentrated:0.25", "concentrated:0.5", "powerlaw:0.25", "powerlaw:0.5"}
HIGH_ENTROPY = {"concentrated:1.0", "concentrated:0.75", "stale_resident", "strided_sweep",
                "sawtooth", "temporal_phases", "zipf:1.0", "hash_synonym", "powerlaw:0.75"}


def cells(binary, series):
    return d[binary].get(series, {})


print("=" * 78)
print("WARP-COALESCE EFFECT — coal/nocoal GiB/s ratio (>1 = coalesce helps), geomean over N")
print("=" * 78)
for binary in d:
    print(f"\n### {binary} ###")
    for algo, coal_s, nocoal_s in [("GPN", "GPN_coal", "GPN_nocoal"), ("DAC", "DAC_coal", "DAC_nocoal")]:
        cs, ns = cells(binary, coal_s), cells(binary, nocoal_s)
        # per shape, geomean ratio over (sample, N, bins)
        byshape = {}
        for k, cv in cs.items():
            nv = ns.get(k)
            if cv and nv and nv > 0:
                sh = k.split("|", 3)[3]
                byshape.setdefault(sh, []).append(cv / nv)
        print(f"  {algo}: coalesce-on / coalesce-off, per shape (geomean):")
        for sh in sorted(byshape, key=lambda s: -geomean(byshape[s])):
            g = geomean(byshape[sh])
            flag = "  coalesce WINS" if g > 1.05 else ("  coalesce HURTS" if g < 0.95 else "")
            print(f"    {sh:20} {g:5.2f}x{flag}")


print("\n" + "=" * 78)
print("VS BASELINE — geomean speedup over BAS (upstream main), by series & entropy class")
print("=" * 78)
for binary in d:
    bas = cells(binary, "BAS")
    print(f"\n### {binary} ###  (geomean default/main and best-coalesce-choice/main)")
    print(f"  {'series':12} {'all-shapes':>11} {'low-entropy':>12} {'high-entropy':>13}")
    for series in ["DEF", "GPN_coal", "GPN_nocoal", "DAC_coal", "DAC_nocoal"]:
        cs = cells(binary, series)
        allr, lowr, highr = [], [], []
        for k, v in cs.items():
            b = bas.get(k)
            if v and b and b > 0:
                r = v / b
                allr.append(r)
                sh = k.split("|", 3)[3]
                (lowr if sh in LOW_ENTROPY else highr if sh in HIGH_ENTROPY else allr).append(r)
        ga, gl, gh = geomean(allr), geomean(lowr), geomean(highr)
        f = lambda x: f"{x:.2f}x" if x else "  -"
        print(f"  {series:12} {f(ga):>11} {f(gl):>12} {f(gh):>13}")

    # Headline: adaptive = pick the better of coal/nocoal PER CELL, vs always-on (the default).
    for algo, coal_s, nocoal_s in [("GPN", "GPN_coal", "GPN_nocoal"), ("DAC", "DAC_coal", "DAC_nocoal")]:
        cs, ns = cells(binary, coal_s), cells(binary, nocoal_s)
        adaptive_over_on = []
        for k, cv in cs.items():
            nv = ns.get(k)
            if cv and nv and cv > 0:
                adaptive_over_on.append(max(cv, nv) / cv)
        g = geomean(adaptive_over_on)
        if g:
            print(f"  -> {algo}: per-cell-adaptive coalesce would gain geomean {g:.2f}x over always-on "
                  f"(max cell {max(adaptive_over_on):.2f}x)")
