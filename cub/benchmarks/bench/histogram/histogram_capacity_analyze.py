#!/usr/bin/env python3
# Analyze + visualize the SMEM capacity exploration:
#   Sweep A (cache_slots_results.json): GiB/s vs forced cache slot count, per shape.
#   Sweep B (priv_cap_results.json):    GiB/s vs privatized-SMEM cap, priv_dynamic
#                                       vs the high-bin algos it replaces.
# Writes figures to figs/ and prints a geomean + win/regression summary so the
# verdict is "does spending SMEM (and losing occupancy) pay off, and where".
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read the capacity-sweep JSONs from / write figs to $HIST_SWEEP_OUTDIR (default:
# cwd) -- this tracked script must not read or write inside the source tree.
DATADIR = os.environ.get("HIST_SWEEP_OUTDIR", ".")
FIGS = os.path.join(DATADIR, "capacity_analyze_figs")
os.makedirs(FIGS, exist_ok=True)

# B200: 228 KB SMEM/SM. blocks/SM = floor(228KB / per-block-smem).
def blocks_per_sm(kb_per_block):
    return max(1, int(233472 / (kb_per_block * 1024)))


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return float(np.exp(np.mean(np.log(xs)))) if xs else 0.0


# ---------- Sweep A: cache slot ladder ----------
def analyze_A():
    p = os.path.join(DATADIR, "cache_slots_results.json")
    if not os.path.exists(p):
        print("[A] no results yet"); return
    data = json.load(open(p))
    print("\n==================== SWEEP A: cache slot count ====================")
    # For each binary+algo, geomean GiB/s across cells per requested-slots, but ONLY
    # over cells that actually ran the forced algo at the requested (launched) slots.
    for label, av in data.items():
        # group tags by algo
        by_algo = defaultdict(dict)
        for tag, rec in av.items():
            algo, slots = tag.split("@")
            by_algo[algo][slots] = rec
        for algo, ladder in by_algo.items():
            # common cell set across slot settings that ran_forced
            valid = {s: r for s, r in ladder.items() if r["ran_forced"] and r["cells"]}
            if not valid:
                continue
            common = set.intersection(*[set(r["cells"]) for r in valid.values()]) if valid else set()
            order = sorted(valid.keys(), key=lambda s: (0 if s == "auto" else int(s)))
            print(f"\n  [{label} / {algo}]  (geomean over {len(common)} common cells)")
            base = None
            for s in order:
                r = valid[s]
                gm = geomean([r["cells"][k] for k in common])
                ls = r["launched_slots"]
                smemkb = r["launched_smem_bytes"] / 1024.0
                bps = blocks_per_sm(max(smemkb, 0.001)) if smemkb else "-"
                if s == "auto":
                    base = gm
                rel = f"{gm/base*100:5.1f}%" if base else "   -  "
                print(f"    req={s:>6} launched_slots={ls:>6} smem={smemkb:6.0f}KB ~{bps}blk/SM  geomean={gm:7.1f} GiB/s  vs_auto={rel}")
            # N-stratified: does a bigger cache help at SMALL N (unsaturated grid)
            # even if it hurts at large N? Compare the largest forced slots vs auto,
            # per element count.
            big = max(valid.keys(), key=lambda s: valid[s]["launched_slots"])
            if "auto" in valid and big != "auto":
                per_n = defaultdict(lambda: [[], []])  # N -> [auto[], big[]]
                for k in common:
                    n = int(k.split("|")[1])
                    per_n[n][0].append(valid["auto"]["cells"][k])
                    per_n[n][1].append(valid[big]["cells"][k])
                print(f"      N-stratified  auto({valid['auto']['launched_slots']}) vs big({valid[big]['launched_slots']}):")
                for n in sorted(per_n):
                    a, b = geomean(per_n[n][0]), geomean(per_n[n][1])
                    print(f"        N={n:>11}: auto={a:7.1f}  big={b:7.1f}  big/auto={b/a*100 if a else 0:5.1f}%")
    _plot_A(data)


def _plot_A(data):
    """Throughput vs cache SIZE, plotted RELATIVE to the status-quo auto sizer
    (= 100%, horizontal reference line) so "more capacity vs shipped" is explicit.

    Fixes: (1) dedupe by LAUNCHED slot count -- single-channel floor is 4096, so
    requests 1024/2048/4096 all clamp to 4096 and must not overplot as 3 points;
    (2) a config that cannot grow (multi-channel: 1024 slots is the only size that
    fits the SMEM opt-in) is drawn as a single annotated marker, not a 1-point line
    masquerading as a sweep; (3) occupancy (blocks/SM) annotated at each point."""
    for label, av in data.items():
        by_algo = defaultdict(dict)
        for tag, rec in av.items():
            algo, slots = tag.split("@")
            by_algo[algo][slots] = rec
        fig, ax = plt.subplots(figsize=(9.5, 6))
        ax.axhline(100.0, color="black", lw=1.2, ls="--", alpha=0.7,
                   label="status quo (auto sizer) = 100%")
        any_multi_point = False
        for algo, ladder in by_algo.items():
            valid = {s: r for s, r in ladder.items() if r["ran_forced"] and r["cells"]}
            if "auto" not in valid:
                continue
            common = set.intersection(*[set(r["cells"]) for r in valid.values()])
            if not common:
                continue
            base = geomean([valid["auto"]["cells"][k] for k in common])
            auto_slots = valid["auto"]["launched_slots"]
            # Dedupe forced runs by the slot count that ACTUALLY launched; for each
            # distinct launched size keep one (prefer the run whose request == launched,
            # else the smallest request that reached it).
            by_launched = {}
            for s, r in valid.items():
                if s == "auto":
                    continue
                ls = int(r["launched_slots"])
                req = int(s)
                prefer = (req == ls)
                if ls not in by_launched or (prefer and not by_launched[ls][0]):
                    by_launched[ls] = (prefer, r)
            pts = sorted(by_launched.items())
            xs = [ls for ls, _ in pts]
            ys = [geomean([r["cells"][k] for k in common]) / base * 100.0 for _, (_, r) in pts]
            if len(xs) >= 2:
                line, = ax.plot(xs, ys, marker="o", label=f"{algo}")
                # annotate occupancy from the recorded launched smem bytes
                for ls, (_, r) in pts:
                    smemkb = r["launched_smem_bytes"] / 1024.0
                    yy = geomean([r["cells"][k] for k in common]) / base * 100.0
                    ax.annotate(f"{smemkb:.0f}KB\n{blocks_per_sm(max(smemkb,0.001))}blk/SM",
                                (ls, yy), textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=7, color=line.get_color())
            elif len(xs) == 1:
                # floor-bound config: a single reachable size. Mark it, don't fake a line.
                any_multi_point = True
                ax.scatter(xs, ys, s=70, marker="D", zorder=5, label=f"{algo} (only reachable size)")
                r = pts[0][1][1]
                smemkb = r["launched_smem_bytes"] / 1024.0
                ax.annotate(f"{xs[0]} slots = {smemkb:.0f}KB ({blocks_per_sm(max(smemkb,0.001))}blk/SM)\n"
                            f"larger sizes exceed the {233472//1024}KB SMEM opt-in -> cannot launch",
                            (xs[0], ys[0]), textcoords="offset points", xytext=(10, -4),
                            ha="left", va="center", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("cache slots / channel (actually launched)")
        ax.set_ylabel("throughput vs status-quo auto sizer (%)")
        chan = "multi-channel" if "multi" in label else "single-channel"
        ax.set_title(f"Lever 1 — high-bin SMEM cache size vs throughput [{label}, {chan}]\n"
                     f"(relative to the shipped occupancy-preserving auto size; >100% = bigger cache wins)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"A_cache_slots_{label}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


# ---------- Sweep B: privatized cap ----------
def analyze_B():
    p = os.path.join(DATADIR, "priv_cap_results.json")
    if not os.path.exists(p):
        print("[B] no results yet"); return
    data = json.load(open(p))
    print("\n==================== SWEEP B: privatized SMEM cap ====================")
    # For each (label, bin, sample, N, shape) we want: priv_dynamic GiB/s at each
    # cap vs the best high-bin alternative. Reorganize by cell.
    # cell key = sample|N|bins|shape ; we compare priv_dynamic (per cap) vs the
    # max over {cuckoo,single_probe,gather,hybrid} (cap-independent baseline).
    # priv_dynamic GiB/s per cell per cap comes from the NATURAL-selection runs, but
    # ONLY for cells where smem_priv_dynamic actually launched (the launch log per
    # run tells us the set of kernels; we approximate per-cell by requiring the run
    # launched priv_dynamic AND the cell's bin <= cap). High-bin alternatives come
    # from the cap16384 forced runs. cell key = label|sample|N|bins|shape.
    priv = defaultdict(dict)     # cell -> {cap: gibs}
    highbin = defaultdict(dict)  # cell -> {algo: gibs}
    capset = set()
    for tag, rec in data.items():
        cap, label, algo = rec["cap"], rec["label"], rec["algo"]
        capset.add(cap)
        for cell, gibs in rec["cells"].items():
            bins = int(cell.split("|")[2])
            ck = f"{label}|{cell}"
            if algo == "natural":
                # natural selection picked priv_dynamic iff bins<=cap AND the run
                # logged it. Record only those (so we compare the on-chip kernel).
                if bins <= cap and "smem_priv_dynamic" in rec["launched"]:
                    priv[ck][cap] = gibs
            elif cap == 16384:  # forced high-bin alternative
                highbin[ck][algo] = max(highbin[ck].get(algo, 0), gibs)
    caps = sorted(capset)
    # Summary: for each cap, geomean of priv_dynamic over cells where it launched,
    # and the geomean of the high-bin best over those same cells.
    print(f"\n  priv_dynamic geomean by cap, vs high-bin best (same cells):")
    for cap in caps:
        cells = [ck for ck in priv if cap in priv[ck]]
        if not cells:
            continue
        gm_priv = geomean([priv[ck][cap] for ck in cells])
        gm_high = geomean([max(highbin[ck].values()) for ck in cells if highbin.get(ck)])
        kb = cap * 4 / 1024.0
        print(f"    cap={cap:>5} ({kb:3.0f}KB, ~{blocks_per_sm(kb)}blk/SM): priv_dynamic={gm_priv:7.1f}  high-bin_best={gm_high:7.1f}  "
              f"priv/high={ (gm_priv/gm_high*100 if gm_high else 0):5.1f}%  over {len(cells)} cells")
    # Win/regression count at each cap: priv_dynamic vs high-bin best per cell.
    print(f"\n  per-cell wins (priv_dynamic > high-bin best) by cap:")
    for cap in caps:
        cells = [ck for ck in priv if cap in priv[ck] and highbin.get(ck)]
        wins = sum(1 for ck in cells if priv[ck][cap] > max(highbin[ck].values()))
        print(f"    cap={cap:>5}: {wins}/{len(cells)} cells where priv_dynamic beats the high-bin path it replaces")

    # Breakdowns: where does the bigger cap win vs lose? Split priv/high ratio by
    # channel(single|multi), by N, and by shape -- at each cap.
    def ratio_geomean(cells, cap):
        rs = [priv[ck][cap] / max(highbin[ck].values()) for ck in cells
              if cap in priv[ck] and highbin.get(ck)]
        return geomean(rs) * 100 if rs else 0.0, len(rs)

    def split(keyfn, title):
        print(f"\n  priv/high % by {title} (>100 = bigger cap wins):")
        groups = defaultdict(list)
        for ck in priv:
            groups[keyfn(ck)].append(ck)
        for cap in caps:
            if cap == 16384:
                continue
            parts = []
            for gkey in sorted(groups):
                r, n = ratio_geomean(groups[gkey], cap)
                if n:
                    parts.append(f"{gkey}={r:5.1f}%({n})")
            print(f"    cap={cap:>5}: " + "  ".join(parts))

    split(lambda ck: "multi" if ck.startswith("multi") else "single", "channel")
    split(lambda ck: f"N={int(ck.split('|')[2]):>11}", "input size N")
    split(lambda ck: ck.split("|")[4], "shape")
    split(lambda ck: f"bins={int(ck.split('|')[3])}", "bin count")
    _plot_B(priv, highbin, caps)


def _plot_B(priv, highbin, caps):
    """Privatized cap RELATIVE to the status quo: each cap's on-chip priv_dynamic
    vs the high-bin path the shipped 16384-cap falls through to at the same cell
    (per-cell geomean ratio). 100% line = status quo; >100% = the bigger cap wins.
    Overlaid: the per-shape ratio, so the hash_synonym regression vs the
    powerlaw/zipf wins is visible, not hidden in an aggregate."""
    caps_big = [c for c in caps if c != 16384]
    if not caps_big:
        plt.close("all"); return
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axhline(100.0, color="black", lw=1.3, ls="--", alpha=0.8, label="status quo (shipped 16384 cap) = 100%")

    def ratios(cells, cap):
        rs = [priv[ck][cap] / max(highbin[ck].values()) for ck in cells
              if cap in priv[ck] and highbin.get(ck)]
        return geomean(rs) * 100 if rs else None

    # overall (single-channel; multi can't launch priv_dynamic at these bins)
    allcells = list(priv.keys())
    overall = [ratios(allcells, c) for c in caps_big]
    ax.plot(caps_big, overall, marker="o", lw=2.6, color="#111", zorder=6, label="overall (single-channel)")
    for c, y in zip(caps_big, overall):
        if y is None:
            continue
        kb = c * 4 / 1024.0
        ax.annotate(f"{y:.1f}%\n{kb:.0f}KB·{blocks_per_sm(kb)}blk/SM", (c, y),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#111")
    # per-shape overlays
    shapes = sorted({ck.split("|")[4] for ck in priv})
    for sh in shapes:
        cells = [ck for ck in priv if ck.split("|")[4] == sh]
        ys = [ratios(cells, c) for c in caps_big]
        if any(y is not None for y in ys):
            ax.plot(caps_big, ys, marker=".", lw=1.0, alpha=0.75, label=sh)
    ax.set_xscale("log", base=2)
    ax.set_xticks(caps_big)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax.set_xlabel("privatized SMEM cap (bins) — capacity increasing →")
    ax.set_ylabel("priv_dynamic throughput vs status-quo high-bin path (%)")
    ax.set_title("Lever 2 — larger privatized SMEM cap vs the status quo it replaces\n"
                 "(per-cell geomean ratio; >100% = the bigger on-chip histogram wins, despite lower occupancy)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "B_priv_cap_vs_statusquo.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    analyze_A()
    analyze_B()
    print(f"\nFigures -> {FIGS}")
