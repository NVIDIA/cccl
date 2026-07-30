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
    Y = GiB/s, one connect-the-dots line per algorithm present in the data
    (markers = measured points, no fitted line). Two reference series stand out:
    `default` (the shipping selector's pick, thick black) and `main` (upstream
    main's default dispatch, dashed grey) -- the baseline this branch improves on.

The `main` series is only drawn for InputShape generators that are byte-identical
between upstream main and this branch (the sweep driver omits it elsewhere, since
a reweighted/ reordered generator would not be an apples-to-apples comparison).

The sweep JSON is produced by `histogram_algo_sweep.py` (the force-hook +
upstream-main sweep driver); see the accompanying README. Run with a Python that
has numpy + matplotlib:
  python histogram_algo_perf.py --results sweep_results.json --outdir algo_perf_figs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import matplotlib
import numpy as np

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

# Algorithms + fixed colors/markers/linestyles (consistent across every plot).
# Names match the post-rework algorithm enum / the CUB_HISTO_FORCE_ALGO values
# the sweep driver (histogram_algo_sweep.py) forces, plus two reference series:
#   default -- the shipping selector's own pick (select_algorithm), drawn as a
#              thick black line so "what CUB actually does" stands out.
#   main    -- upstream `main`'s default dispatch (the baseline this branch
#              improves on), drawn dashed grey. Only present for InputShape
#              generators that are byte-identical between main and this branch
#              (see MAIN_COMPARABLE_SHAPES in the sweep driver).
# (color, marker, label, linestyle, linewidth)
# The two reference series are deliberately the most prominent marks on the plot:
# `default` = thick solid black (what CUB ships), `main` = thick bright-red
# dash-dot with big X markers (the upstream baseline we beat). They are drawn LAST
# (highest zorder) so they sit on top of the candidate-algorithm cluster instead of
# being buried under it.
ALGO_STYLE = {
    "default": ("#000000", "*", "selector default (ships)", "-", 3.4),
    "main": ("#e6194B", "X", "UPSTREAM main (default)", "-.", 3.4),
    "gmem_privatized_nocache": (
        "#9467bd",
        "^",
        "gmem-priv gather (no cache)",
        "-",
        1.6,
    ),
    "gmem_privatized_cuckoo": ("#1f77b4", "o", "gmem-priv + cuckoo", "-", 1.6),
    "gmem_privatized_single_probe": (
        "#17becf",
        "s",
        "gmem-priv + single-probe",
        "-",
        1.6,
    ),
    "direct_cuckoo": ("#2ca02c", "o", "direct-atomic + cuckoo", "-", 1.6),
    "direct_single_probe": ("#8c8c00", "s", "direct-atomic + single-probe", "-", 1.6),
    "direct_nocache": ("#ff7f0e", "v", "direct atomics (no cache)", "-", 1.6),
}
# Draw order: reference series last (on top). gmem/direct candidates first.
ALGO_ORDER = [
    "gmem_privatized_nocache",
    "gmem_privatized_cuckoo",
    "gmem_privatized_single_probe",
    "direct_cuckoo",
    "direct_single_probe",
    "direct_nocache",
    "main",
    "default",
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
    if e >= 1 << 30:
        return f"{e / (1 << 30):.2f}G"  # non-power-of-2 (e.g. 2e9 -> 1.86G)
    if e >= 1 << 20:
        return f"{e / (1 << 20):.1f}M"
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
            if (
                s == sample
                and int(e) == elements
                and sh == shape
                and int(b) >= MIN_PLOT_BINS
            ):
                pts.append((int(b), gibs))
        pts.sort()
        if pts:
            out[algo] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def draw_perf(ax, series, title):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("# bins")
    ax.set_ylabel("GiB/s (log)")
    ax.set_xscale("log", base=2)
    # Log y: the upstream `main` baseline can be 30x slower than the branch at high
    # bins; on a linear axis it is pinned to ~0 and unreadable. Log y keeps the slow
    # baseline visible AND makes the multiplicative speedup (vertical gap) the
    # eye-level quantity.
    ax.set_yscale("log")
    any_pts = False
    drawn = [a for a in ALGO_ORDER if a in series and len(series[a][0])]
    # cuckoo / single-probe / no-cache often land on nearly the same curve, so each
    # series carries its own color/marker/linestyle/linewidth (the reference series
    # -- default, main -- are wider/dashed and drawn last so they sit on top).
    for i, algo in enumerate(drawn):
        color, marker, label, ls, lw = ALGO_STYLE[algo]
        xb, yv = series[algo]
        any_pts = True
        # Reference series (default, main) get larger markers, full opacity, and the
        # highest zorder so they read clearly over the candidate-algorithm cluster.
        is_ref = algo in ("default", "main")
        ax.plot(
            xb,
            yv,
            color=color,
            marker=marker,
            markersize=10 if is_ref else 6,
            lw=lw,
            linestyle=ls,
            alpha=1.0 if is_ref else 0.8,
            label=label,
            zorder=(20 if is_ref else 3 + i),
        )
    # Shade the speedup region between the upstream `main` baseline and the shipping
    # `default` so the gap reads even at small panel size (and where main's thin line
    # would otherwise hug the bottom). Only where both series share bin points.
    if "main" in series and "default" in series:
        mb, mv = series["main"]
        db, dv = series["default"]
        common = sorted(set(int(x) for x in mb) & set(int(x) for x in db))
        if common:
            mmap = {int(x): y for x, y in zip(mb, mv)}
            dmap = {int(x): y for x, y in zip(db, dv)}
            xs = np.array(common)
            lo = np.array([mmap[x] for x in common])
            hi = np.array([dmap[x] for x in common])
            ax.fill_between(
                xs,
                lo,
                hi,
                where=(hi >= lo),
                color="#e6194B",
                alpha=0.08,
                zorder=1,
                label="_nolegend_",
            )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xticks(sorted({int(b) for s in series.values() for b in s[0]}))
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda v, _: fmt_bins(int(round(v))))
    )
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    # log y: leave matplotlib's autoscaled positive limits (a bottom=0 is invalid).
    return any_pts


def draw_hitrate(ax, hr_algo_cells, shape, elements_list, title):
    """Cache hit rate (%) vs #bins, one series per #elements. hr_algo_cells is
    keyed 'Bins|Elements|InputShape' -> {rate,hits,misses}."""
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("# bins")
    ax.set_ylabel("cache hit rate (%)")
    ax.set_xscale("log", base=2)
    any_pts = False
    all_bins = set()
    cmap = plt.cm.viridis
    for i, elements in enumerate(elements_list):
        pts = []
        for key, rec in hr_algo_cells.items():
            b, e, sh = key.split("|")
            if int(e) == elements and sh == shape:
                pts.append((int(b), rec["rate"] * 100.0))
        pts.sort()
        if pts:
            any_pts = True
            xb = [p[0] for p in pts]
            yv = [p[1] for p in pts]
            all_bins.update(xb)
            ax.plot(
                xb,
                yv,
                marker="o",
                ms=5,
                lw=1.8,
                color=cmap(0.1 + 0.8 * i / max(1, len(elements_list) - 1)),
                label=fmt_elements(elements),
            )
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if all_bins:
        ax.set_xticks(sorted(all_bins))
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda v, _: fmt_bins(int(round(v))))
        )
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    if any_pts:
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7, title="# elements", title_fontsize=7)
    else:
        ax.text(
            0.5,
            0.5,
            "no hit-rate data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
    return any_pts


def render_one(
    binary_label,
    per_algo_cells,
    sample,
    shape,
    elements_list,
    algos,
    outpath,
    hr_for_binary=None,
):
    """One PNG: characterization top row + per-element-count perf grid, and (when
    hit-rate data is supplied) a final row with cuckoo and single-probe cache
    hit-rate vs #bins (one series per #elements)."""
    bins, counts, char_bins = C.char_input(shape)
    hr_for_binary = hr_for_binary or {}
    has_hr = bool(
        hr_for_binary.get("direct_cuckoo") or hr_for_binary.get("direct_single_probe")
    )

    ncols = 3
    nperf = len(elements_list)
    perf_rows = (nperf + ncols - 1) // ncols
    nrows = 1 + perf_rows + (1 if has_hr else 0)

    fig = plt.figure(figsize=(5.4 * ncols, 4.1 * nrows))
    gs = fig.add_gridspec(nrows, ncols)

    transform, channels = BINARY_META[binary_label]
    head = f"{transform.upper()} · {channels}-channel · {sample}  —  InputShape: {shape}  ({C.SHAPE_BLURB.get(shape, '')})"
    head = "\n".join(textwrap.wrap(head, width=120))
    hr_note = (
        "   bottom row: SMEM-cache hit rate vs #bins (series = #elements)"
        if has_hr
        else ""
    )
    fig.suptitle(
        f"{head}\n"
        f"top: input characterization (N={C.fmt_int(C.CHAR_N)}, bins={C.fmt_bins(char_bins)})   "
        f"middle: GiB/s vs #bins per input size — one line per algorithm{hr_note}",
        fontsize=12,
    )

    C.draw_distribution(fig.add_subplot(gs[0, 0]), counts, char_bins)
    C.draw_sequence(fig.add_subplot(gs[0, 1]), bins, char_bins, shape=shape)
    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis("off")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=ALGO_STYLE[a][0],
            marker=ALGO_STYLE[a][1],
            linestyle=ALGO_STYLE[a][3],
            lw=ALGO_STYLE[a][4],
            label=ALGO_STYLE[a][2],
        )
        for a in algos
    ]
    legend_ax.legend(
        handles=handles, loc="center", fontsize=11, title="algorithms", frameon=True
    )

    for i, elements in enumerate(elements_list):
        r, c = 1 + i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        series = perf_series(per_algo_cells, sample, elements, shape, algos)
        if not draw_perf(ax, series, f"N = {fmt_elements(elements)} elements"):
            ax.text(
                0.5,
                0.5,
                "no data\n(all cells skipped)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )

    if has_hr:
        hr_row = 1 + perf_rows
        draw_hitrate(
            fig.add_subplot(gs[hr_row, 0]),
            hr_for_binary.get("direct_cuckoo", {}),
            shape,
            elements_list,
            "cuckoo cache — hit rate vs #bins",
        )
        draw_hitrate(
            fig.add_subplot(gs[hr_row, 1]),
            hr_for_binary.get("direct_single_probe", {}),
            shape,
            elements_list,
            "single-probe cache — hit rate vs #bins",
        )
        # third column of the hit-rate row: short explainer
        ax = fig.add_subplot(gs[hr_row, 2])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "hit = contribution absorbed in the\nSMEM cache (block-scope add)\n"
            "miss = spilled to a GMEM atomic\n\n(hit rate is sample-type\nindependent; measured on a\n"
            "separate instrumented build)",
            ha="center",
            va="center",
            fontsize=9,
            color="#333",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results",
        default="sweep_results_6shape.json",
        help="per-cell perf sweep JSON (binary -> algo -> 'SampleT|Elements|Bins|InputShape')",
    )
    ap.add_argument(
        "--hitrate",
        default="hitrate_results.json",
        help="per-cell hit-rate sweep JSON (binary -> algo -> 'Bins|Elements|InputShape' -> {rate}); "
        "optional, adds two cache-hit-rate panels per image when present",
    )
    ap.add_argument(
        "--outdir",
        default="algo_perf_figs",
        help="output directory for the per-shape PNGs",
    )
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(f"missing results JSON: {args.results} (pass --results)")
    data = json.load(open(args.results))
    hitrate = {}
    if args.hitrate and os.path.exists(args.hitrate):
        hitrate = json.load(open(args.hitrate))
        print(f"hit-rate data: {args.hitrate}")
    else:
        print(f"(no hit-rate JSON at {args.hitrate}; hit-rate panels skipped)")

    os.makedirs(args.outdir, exist_ok=True)
    written = []
    for binary_label, per_algo_cells in data.items():
        transform, channels = BINARY_META[binary_label]
        # Plot every algorithm series actually present in the data for this binary
        # (in canonical ALGO_ORDER). `main` appears only for the generator-identical
        # shapes (the sweep driver omits it elsewhere); a forced algo absent at a
        # given (transform, channels) simply has no points and is dropped per panel.
        algos = [a for a in ALGO_ORDER if a in per_algo_cells]

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
            shapes = sorted(
                {k.split("|")[3] for cells in per_algo_cells.values() for k in cells},
                key=lambda s: (C.SHAPES.index(s) if s in C.SHAPES else 999, s),
            )
            hr_for_binary = hitrate.get(binary_label, {})
            for shape in shapes:
                outpath = os.path.join(folder, f"{shape.replace(':', '_')}.png")
                render_one(
                    binary_label,
                    per_algo_cells,
                    sample,
                    shape,
                    elements,
                    algos,
                    outpath,
                    hr_for_binary,
                )
                written.append(outpath)

    print(f"Wrote {len(written)} images under {args.outdir}")


if __name__ == "__main__":
    main()
