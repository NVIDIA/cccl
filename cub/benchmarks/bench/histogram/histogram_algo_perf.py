#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Performance comparison of the high-bin histogram algorithm candidates.

Consumes a per-cell sweep JSON (each high-bin algorithm forced across
SampleT x Elements x Bins x InputShape, per benchmark binary) and the
source-of-truth input generators in `histogram_input_design.py`.

Layout (one image per InputShape, inside a folder per transform/channels/type):
  algo_perf_figs/<even|range>_<single|multi>_<I32|F64>/<input_shape>.png
Each image:
  * top row : input characterization for that shape -- distribution (log-y) and
    position-in-sequence -- drawn by the SHARED functions in
    histogram_input_characterization.py (so the characterization here matches the
    standalone characterization figures exactly), plus an algorithm legend.
  * below   : one performance graph per #input-elements; X = #bins (log2),
    Y = GiB/s, one connect-the-dots line per algorithm valid for this
    (transform, channels) combination (markers = measured points, no fitted line).

`hybrid_single_pass` is single-channel-only, so it is omitted from the
multi-channel folders.

The sweep JSON is produced by the (scratch, force-hook) sweep driver; see the
accompanying README. Run with a Python that has numpy + matplotlib:
  python histogram_algo_perf.py --results sweep_results.json --outdir algo_perf_figs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import histogram_input_characterization as C  # noqa: E402  (shared draw_* + char_input)

# binary label in the sweep JSON -> (transform, channels)
BINARY_META = {
    "even": ("even", "single"),
    "range": ("range", "single"),
    "multi_even": ("even", "multi"),
    "multi_range": ("range", "multi"),
}

# Algorithms + fixed colors/markers (consistent across every plot).
ALGO_STYLE = {
    "direct_atomic_cuckoo": ("#1f77b4", "o", "cuckoo cache"),
    "direct_atomic_single_probe": ("#2ca02c", "s", "single-probe cache"),
    "direct_atomic_no_cache": ("#ff7f0e", "v", "no cache (direct atomics)"),
    "gmem_priv_gather": ("#9467bd", "^", "gmem gather-merge"),
    "hybrid_single_pass": ("#d62728", "D", "hybrid SMEM+GMEM"),
}
ALGO_ORDER = [
    "direct_atomic_cuckoo",
    "direct_atomic_single_probe",
    "direct_atomic_no_cache",
    "gmem_priv_gather",
    "hybrid_single_pass",
]

# Plot the whole swept bin range. The force harness overrides both dispatch
# gates (direct-atomic bin threshold and the hybrid kSplitBin guard) and every
# forced run is launch-validated, so each algorithm -- including hybrid -- is
# genuinely measured down to the smallest swept bin count.
MIN_PLOT_BINS = 0


def fmt_elements(e: int) -> str:
    e = int(e)
    if e % (1 << 30) == 0:
        return f"{e >> 30}G"
    if e % (1 << 20) == 0:
        return f"{e >> 20}M"
    return str(e)


def fmt_bins(b: int) -> str:
    b = int(b)
    return f"{b // 1024}K" if b % 1024 == 0 else str(b)


def perf_series(per_algo_cells, sample, elements, shape, algos):
    """For a fixed (sample, elements, shape): {algo: (bins[], gibs[])}."""
    out = {}
    for algo in algos:
        cells = per_algo_cells.get(algo, {})
        pts = []
        for key, gibs in cells.items():
            s, e, b, sh = key.split("|")
            if s == sample and int(e) == elements and sh == shape and int(b) >= MIN_PLOT_BINS:
                pts.append((int(b), gibs))
        pts.sort()
        if pts:
            out[algo] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def draw_perf(ax, series, title):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("# bins")
    ax.set_ylabel("GiB/s")
    ax.set_xscale("log", base=2)
    any_pts = False
    drawn = [a for a in ALGO_ORDER if a in series and len(series[a][0])]
    # cuckoo / single-probe / no-cache often land on nearly the same curve: taper
    # linewidth across draw order (earlier = wider, underneath) and keep lines
    # semi-transparent + distinctly marked so every series stays visible.
    for i, algo in enumerate(drawn):
        color, marker, label = ALGO_STYLE[algo]
        xb, yv = series[algo]
        any_pts = True
        lw = 3.0 - 0.5 * i
        ax.plot(xb, yv, color=color, marker=marker, markersize=6, lw=max(lw, 1.0),
                alpha=0.75, label=label, zorder=3 + i)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xticks(sorted({int(b) for s in series.values() for b in s[0]}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_bins(int(round(v)))))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    if any_pts:
        ax.set_ylim(bottom=0)
    return any_pts


def render_one(binary_label, per_algo_cells, sample, shape, elements_list, algos, outpath):
    """One PNG: characterization top row + per-element-count perf grid."""
    bins, counts, char_bins = C.char_input(shape)

    ncols = 3
    nperf = len(elements_list)
    perf_rows = (nperf + ncols - 1) // ncols
    nrows = 1 + perf_rows

    fig = plt.figure(figsize=(5.4 * ncols, 4.1 * nrows))
    gs = fig.add_gridspec(nrows, ncols)

    transform, channels = BINARY_META[binary_label]
    fig.suptitle(
        f"{transform.upper()} · {channels}-channel · {sample}  —  InputShape: {shape}"
        f"  ({C.SHAPE_BLURB.get(shape, '')})\n"
        f"top: input characterization (N={C.fmt_int(C.CHAR_N)}, bins={C.fmt_bins(char_bins)})   "
        f"below: GiB/s vs #bins per input size — one line per algorithm (markers = measured points)",
        fontsize=12,
    )

    C.draw_distribution(fig.add_subplot(gs[0, 0]), counts, char_bins)
    C.draw_sequence(fig.add_subplot(gs[0, 1]), bins, char_bins)
    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=ALGO_STYLE[a][0], marker=ALGO_STYLE[a][1], lw=1.5, label=ALGO_STYLE[a][2])
        for a in algos
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=11, title="algorithms", frameon=True)

    for i, elements in enumerate(elements_list):
        r, c = 1 + i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        series = perf_series(per_algo_cells, sample, elements, shape, algos)
        if not draw_perf(ax, series, f"N = {fmt_elements(elements)} elements"):
            ax.text(0.5, 0.5, "no data\n(all cells skipped)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="sweep_results_6shape.json",
                    help="per-cell sweep JSON (keyed by binary -> algo -> 'SampleT|Elements|Bins|InputShape')")
    ap.add_argument("--outdir", default="algo_perf_figs", help="output directory for the per-shape PNGs")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"missing results JSON: {args.results} (pass --results)")
    data = json.load(open(args.results))

    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for binary_label, per_algo_cells in data.items():
        transform, channels = BINARY_META[binary_label]
        algos = list(ALGO_ORDER) if channels == "single" else [a for a in ALGO_ORDER if a != "hybrid_single_pass"]

        samples, elements = set(), set()
        for cells in per_algo_cells.values():
            for key in cells:
                s, e, _, _ = key.split("|")
                samples.add(s)
                elements.add(int(e))
        elements = sorted(elements)

        for sample in sorted(samples):
            folder = os.path.join(args.outdir, f"{transform}_{channels}_{sample}")
            os.makedirs(folder, exist_ok=True)
            # Render every shape present in the data for this binary.
            shapes = sorted({k.split("|")[3] for cells in per_algo_cells.values() for k in cells},
                            key=lambda s: (C.SHAPES.index(s) if s in C.SHAPES else 999, s))
            for shape in shapes:
                outpath = os.path.join(folder, f"{shape.replace(':', '_')}.png")
                render_one(binary_label, per_algo_cells, sample, shape, elements, algos, outpath)
                written.append(outpath)

    print(f"Wrote {len(written)} images under {args.outdir}")


if __name__ == "__main__":
    main()
