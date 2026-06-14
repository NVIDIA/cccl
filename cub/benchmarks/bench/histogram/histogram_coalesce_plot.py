#!/usr/bin/env python3
# Plot the targeted warp-coalesce sweep (coalesce_sweep.json). Two figure families:
#
#  A) speedup-vs-#bins:  per (binary, sample, shape) one PNG with a panel per element
#     count; X = #bins (log2), Y = GiB/s ratio over BAS (upstream main, log2, symmetric
#     about 1x). Six lines: DEF, GPN_coal, GPN_nocoal, DAC_coal, DAC_nocoal (BAS is the
#     1x reference). Shows where each coalesce variant beats main and the selector.
#
#  B) coal-vs-nocoal-by-shape:  per (binary, sample) one PNG, a panel per element count;
#     X = #bins, Y = coalesce-on / coalesce-off ratio (log2, 1x line). One line per shape,
#     colored by entropy class. Directly answers "does coalescing help or hurt, by input".
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("coalesce_sweep.json")
outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else src.parent / "figs"
d = json.load(open(src))

# Series styles (color, marker, linestyle, label). BAS is the 1x reference (not drawn).
STYLE = {
    "DEF":        ("black",   "o", "-",  2.2, "DEF — selector default"),
    "GPN_coal":   ("#1f77b4", "^", "-",  1.6, "GPN + coalesce"),
    "GPN_nocoal": ("#1f77b4", "v", "--", 1.6, "GPN no-coalesce"),
    "DAC_coal":   ("#d62728", "s", "-",  1.6, "DAC + coalesce"),
    "DAC_nocoal": ("#d62728", "D", "--", 1.6, "DAC no-coalesce"),
}
DRAW_SERIES = ["DEF", "GPN_coal", "GPN_nocoal", "DAC_coal", "DAC_nocoal"]

SHAPE_ORDER = ["concentrated:1.0", "concentrated:0.75", "concentrated:0.5", "concentrated:0.25",
               "concentrated:0.0", "powerlaw:0.75", "powerlaw:0.5", "powerlaw:0.25", "zipf:1.0",
               "hash_synonym", "stale_resident", "temporal_phases", "strided_sweep", "sawtooth"]
LOW_ENTROPY = {"concentrated:0.0", "concentrated:0.25", "concentrated:0.5", "powerlaw:0.25", "powerlaw:0.5"}

BINARY_LABEL = {"even": "EVEN (ScaleTransform)", "range": "RANGE (SearchTransform)"}


def axes_of(cells):
    samples, elements, bins, shapes = set(), set(), set(), set()
    for k in cells:
        s, e, b, sh = k.split("|", 3)
        samples.add(s); elements.add(int(e)); bins.add(int(b)); shapes.add(sh)
    return sorted(samples), sorted(elements), sorted(bins), shapes


def fmt_elem(e):
    return f"{e // (1 << 30)}G" if e >= (1 << 30) else f"{e // (1 << 20)}M"


def log2_yaxis(ax, vals):
    ax.set_yscale("log", base=2)
    ax.axhline(1.0, color="gray", lw=1.0, alpha=0.7)
    ax.grid(True, which="both", alpha=0.2)
    # tick labels as plain multiples
    import numpy as np
    finite = [v for v in vals if v and v > 0]
    if not finite:
        return
    lo, hi = min(finite), max(finite)
    tlo, thi = math.floor(math.log2(lo)), math.ceil(math.log2(hi))
    ticks = [2.0 ** t for t in range(tlo, thi + 1)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}×" for t in ticks])


def fig_speedup(binary, sample, shape, elements, bins, cells, outpath):
    ncol = len(elements)
    fig, axs = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    allvals = []
    for ci, e in enumerate(elements):
        ax = axs[0][ci]
        for series in DRAW_SERIES:
            color, mk, ls, lw, _ = STYLE[series]
            xs, ys = [], []
            for b in bins:
                key = f"{sample}|{e}|{b}|{shape}"
                v = cells[series].get(key)
                bas = cells["BAS"].get(key)
                if v and bas and bas > 0:
                    xs.append(b); ys.append(v / bas); allvals.append(v / bas)
            if xs:
                ax.plot(xs, ys, color=color, marker=mk, linestyle=ls, lw=lw, ms=5, label=STYLE[series][4])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("#bins")
        ax.set_title(f"N = {fmt_elem(e)}")
        if ci == 0:
            ax.set_ylabel("speedup vs upstream main (×)")
    for ax in axs[0]:
        log2_yaxis(ax, allvals)
    axs[0][-1].legend(fontsize=8, loc="best")
    fig.suptitle(f"{BINARY_LABEL.get(binary, binary)} · {sample} · {shape}\n"
                 f"warp-coalesce on/off for GPN & DAC vs selector (DEF) — baseline = upstream main (1×)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_ratio_by_shape(binary, sample, elements, bins, cells, outpath):
    """coal/nocoal ratio per shape, one panel per element count."""
    ncol = len(elements)
    fig, axs = plt.subplots(2, ncol, figsize=(4.2 * ncol, 7.2), squeeze=False)
    cmap = plt.get_cmap("viridis")
    allvals = []
    for row, (algo, cs, ns) in enumerate([("GPN", "GPN_coal", "GPN_nocoal"),
                                          ("DAC", "DAC_coal", "DAC_nocoal")]):
        for ci, e in enumerate(elements):
            ax = axs[row][ci]
            for si, shape in enumerate(SHAPE_ORDER):
                xs, ys = [], []
                for b in bins:
                    key = f"{sample}|{e}|{b}|{shape}"
                    cv = cells[cs].get(key); nv = cells[ns].get(key)
                    if cv and nv and nv > 0:
                        xs.append(b); ys.append(cv / nv); allvals.append(cv / nv)
                if xs:
                    lo = shape in LOW_ENTROPY
                    ax.plot(xs, ys, color=cmap(si / len(SHAPE_ORDER)),
                            marker="o" if lo else "x", ms=4, lw=1.3,
                            linestyle="-" if lo else "--", label=shape)
            ax.set_xscale("log", base=2)
            ax.axhline(1.0, color="black", lw=1.2)
            ax.grid(True, which="both", alpha=0.2)
            ax.set_xlabel("#bins")
            if ci == 0:
                ax.set_ylabel(f"{algo}: coalesce-on / off (×)")
            ax.set_title(f"{algo} · N = {fmt_elem(e)}")
    for r in range(2):
        for ax in axs[r]:
            log2_yaxis(ax, allvals)
    axs[0][-1].legend(fontsize=6, loc="upper right", ncol=1)
    fig.suptitle(f"{BINARY_LABEL.get(binary, binary)} · {sample} — warp-coalesce ON÷OFF by input shape\n"
                 f">1 = coalescing helps; <1 = coalescing hurts. solid/circle = low-entropy, dashed/x = high-entropy",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    n = 0
    for binary in d:
        cells = d[binary]
        samples, elements, bins, shapes = axes_of(cells["DEF"])
        for sample in samples:
            # A) per-shape speedup figures
            folder = outdir / f"{binary}_{sample}"
            folder.mkdir(parents=True, exist_ok=True)
            for shape in sorted(shapes, key=lambda s: SHAPE_ORDER.index(s) if s in SHAPE_ORDER else 99):
                fig_speedup(binary, sample, shape, elements, bins, cells,
                            folder / f"{shape.replace(':', '_')}.png")
                n += 1
            # B) the coal/nocoal-by-shape summary figure
            fig_ratio_by_shape(binary, sample, elements, bins, cells,
                               outdir / f"coalesce_ratio_{binary}_{sample}.png")
            n += 1
    print(f"wrote {n} figures under {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
