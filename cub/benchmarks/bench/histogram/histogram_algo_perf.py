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
# Every series has a UNIQUE (color, marker, linestyle) so overlapping curves stay
# distinguishable: the candidate algorithms often collapse onto nearly the same
# curve (e.g. cuckoo vs single-probe), so they must differ in BOTH marker shape and
# dash pattern -- where two lines coincide you then see a dashed line riding over a
# solid one rather than one hiding the other. Markers are drawn semi-transparent and
# on top of all lines so coincident points blend visibly instead of one hiding another.
# (color, marker, label, linestyle, linewidth)
ALGO_STYLE = {
    "default": ("#000000", "*", "selected default", "-", 2.2),
    "main": ("#e6194B", "X", "baseline (upstream main)", "-.", 2.2),
    "gmem_privatized_nocache": ("#9467bd", "^", "gmem-priv + no cache (gather)", "-", 1.1),
    "hybrid": ("#e377c2", "P", "hybrid (SMEM+GMEM single-pass)", "--", 1.1),
    "gmem_privatized_cuckoo": ("#1f77b4", "o", "gmem-priv + cuckoo", "--", 1.1),
    "gmem_privatized_single_probe": ("#17becf", "s", "gmem-priv + single-probe", ":", 1.1),
    "direct_cuckoo": ("#2ca02c", "D", "direct-atomic + cuckoo", "--", 1.1),
    "direct_single_probe": ("#8c8c00", "P", "direct-atomic + single-probe", ":", 1.1),
    "direct_nocache": ("#ff7f0e", "v", "direct-atomic + no cache", "-", 1.1),
    # Privatized-SMEM kernel. The CUB_HISTO_FORCE_ALGO hook does not force it (and
    # gates forcing to the high-bin tier), but it IS what the selector runs at bins
    # <= SMEM_PRIVATIZED_MAX_BINS -- so the plotter synthesizes this series from the `default`
    # values over that range (see perf_series). Drawn so the low-bin region is a
    # named algorithm, not two unlabeled reference lines.
    "smem_privatized": ("#8c564b", "h", "smem privatized", "-", 1.3),
}
# 3-letter tag per algorithm, used to label each point of the `default` series with
# the algorithm the selector actually picked there. `hybrid` is the smem_split>0
# member of the gmem_privatized_nocache kernel; the dispatch launch tag reports it as
# `gmem_privatized_nocache:hybrid`, which the sweep stores verbatim, so map both the
# member-suffixed name and the bare `hybrid` here.
ALGO_TAG = {
    "smem_privatized": "SMP",
    "gmem_privatized_nocache": "GPN",
    "gmem_privatized_nocache:hybrid": "HYB",
    "hybrid": "HYB",
    "gmem_privatized_cuckoo": "GPC",
    "gmem_privatized_single_probe": "GPS",
    "direct_cuckoo": "DAC",
    "direct_single_probe": "DAS",
    "direct_nocache": "DAN",
    "main": "MAIN",
    "default": "DEF",
}
# Draw order: reference series last (on top). gmem/direct candidates first.
ALGO_ORDER = [
    "smem_privatized",
    "gmem_privatized_nocache",
    "hybrid",
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

# At or below this bin count the whole histogram fits in privatized SMEM, so the
# selector always runs the smem_privatized kernel and the high-bin
# CUB_HISTO_FORCE_ALGO override is a no-op -- the forced gmem-priv / direct-atomic
# series therefore have NO distinct data here (only `default` and `main` are drawn,
# and `default` IS smem_privatized). cub/.../dispatch_histogram.cuh: max_dynamic_smem_bins.
SMEM_PRIVATIZED_MAX_BINS = 16384


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


def selected_algo_tags(per_algo_cells, sample, elements, shape):
    """Which algorithm the selector's `default` actually launched at each bin, for
    labeling the default series. Returns {bins: 3-letter tag}.

    Ground truth: the sweep driver records the selector's pick per cell in the
    `_selected` map (from the dispatch's CUB_HISTO_LOG_LAUNCH tag) -- we read that
    directly, no inference. For older sweep JSONs without `_selected`, fall back to
    matching `default`'s GiB/s to the forced series it equals (smem_privatized below
    the on-chip bin cap)."""
    tags = {}
    selected = per_algo_cells.get("_selected", {})
    if selected:  # ground-truth path
        for key, ran in selected.items():
            s, e, b, sh = key.split("|")
            if s == sample and int(e) == elements and sh == shape and ran in ALGO_TAG:
                tags[int(b)] = ALGO_TAG[ran]
        return tags

    # Fallback for legacy JSON: infer from values.
    deflt = per_algo_cells.get("default", {})
    forced = ["gmem_privatized_nocache", "gmem_privatized_cuckoo", "gmem_privatized_single_probe",
              "direct_cuckoo", "direct_single_probe", "direct_nocache"]
    for key, dv in deflt.items():
        s, e, b, sh = key.split("|")
        if s != sample or int(e) != elements or sh != shape or dv <= 0:
            continue
        bins = int(b)
        if bins <= SMEM_PRIVATIZED_MAX_BINS:
            tags[bins] = ALGO_TAG["smem_privatized"]
            continue
        best, best_err = None, None
        for a in forced:
            av = per_algo_cells.get(a, {}).get(key)
            if av is None or av <= 0:
                continue
            err = abs(av - dv) / dv
            if best_err is None or err < best_err - 1e-9:
                best, best_err = a, err
        if best is not None and best_err is not None and best_err < 0.03:
            tags[bins] = ALGO_TAG[best]
    return tags


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
    # Synthesize the `smem_privatized` series: it is never measured under that name
    # (the force hook neither accepts nor needs it), but it IS exactly what the
    # selector `default` runs for bins <= SMEM_PRIVATIZED_MAX_BINS -- so derive it from the
    # default points over the smem-privatized range. Makes the low-bin region a named
    # algorithm instead of an unexplained `default`/`main`-only stretch. (Only added
    # when not already an explicit series, so a future real measurement would win.)
    if "smem_privatized" in algos and "smem_privatized" not in out and "default" in out:
        db, dv = out["default"]
        mask = [j for j, b in enumerate(db) if int(b) <= SMEM_PRIVATIZED_MAX_BINS]
        if mask:
            out["smem_privatized"] = (np.array([db[j] for j in mask]), np.array([dv[j] for j in mask]))
    return out


def draw_perf(ax, series, title, default_tags=None):
    """Plot speedup-vs-`baseline` (upstream main) for each algorithm: y = (algo GiB/s)
    / (main GiB/s) at each bin count, on a LOG2 y-axis (so 0.5x and 2x sit
    symmetrically about the 1x baseline and a wide speedup range stays legible),
    tick-labelled in plain multiples (1x, 2x, 4x, ...). main is the y=1 baseline line
    and the `default`'s gain is shaded; each default point is annotated with the
    3-letter tag of the algorithm the selector picked there (`default_tags`: {bins:
    tag}). If no baseline is present for this cell, fall back to absolute GiB/s."""
    default_tags = default_tags or {}
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("# bins")
    ax.set_xscale("log", base=2)
    any_pts = False

    # Baseline = upstream main. Map bin -> main GiB/s for this (sample, elements, shape).
    baseline = None
    if "main" in series and len(series["main"][0]):
        mb, mv = series["main"]
        baseline = {int(x): y for x, y in zip(mb, mv) if y > 0}

    if baseline:
        ax.set_yscale("log", base=2)
        ax.set_ylabel("speedup vs baseline (×, log2)")
        # Every algorithm EXCEPT main, divided by main at the shared bin points.
        drawn = [a for a in ALGO_ORDER if a != "main" and a in series and len(series[a][0])]
        for i, algo in enumerate(drawn):
            color, marker, label, ls, lw = ALGO_STYLE[algo]
            xb, yv = series[algo]
            pts = [(int(x), y / baseline[int(x)]) for x, y in zip(xb, yv) if int(x) in baseline]
            if not pts:
                continue
            any_pts = True
            xs = np.array([p[0] for p in pts], dtype=float)
            sp = np.array([p[1] for p in pts])
            is_ref = (algo == "default")
            # Lines drawn first (lower zorder); markers all sit ON TOP (zorder 30+)
            # at the TRUE x, semi-transparent so overlapping points blend visibly
            # (two coincident markers read darker / show both colors) rather than one
            # opaque marker hiding another. Distinct marker shapes + dash patterns
            # still tell coincident series apart; no x-dodge (true data positions).
            # The `default` series is also semi-transparent (both line and marker) so
            # the candidate series it rides on top of remain visible underneath.
            line_alpha = 0.5 if is_ref else 0.85
            mark_alpha = 0.5 if is_ref else 0.6
            ax.plot(xs, sp, color=color, lw=lw, linestyle=ls, marker="",
                    alpha=line_alpha, label=label, zorder=(20 if is_ref else 3 + i))
            ax.plot(xs, sp, color=color, marker=marker, markersize=8 if is_ref else 5.5,
                    linestyle="none", alpha=mark_alpha,
                    zorder=(32 if is_ref else 30 + i))
            # Annotate each default point with the selected algorithm's 3-letter tag.
            if is_ref and default_tags:
                for x, y in zip(xs, sp):
                    tag = default_tags.get(int(round(x)))
                    if tag:
                        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(0, 7),
                                    ha="center", va="bottom", fontsize=6, fontweight="bold",
                                    color="#000000", zorder=40)
        # Baseline reference: main is 1x by definition. Draw it as the styled main line.
        b_color, _, _, b_ls, b_lw = ALGO_STYLE["main"]
        ax.axhline(1.0, color=b_color, linestyle=b_ls, lw=b_lw, alpha=1.0,
                   zorder=19, label="baseline (upstream main), 1×")
        # Shade the shipping default's gain over the baseline (between 1x and default).
        if "default" in series and len(series["default"][0]):
            db, dv = series["default"]
            dpts = sorted((int(x), y / baseline[int(x)]) for x, y in zip(db, dv) if int(x) in baseline)
            if dpts:
                xs = np.array([p[0] for p in dpts])
                hi = np.array([p[1] for p in dpts])
                ax.fill_between(xs, 1.0, hi, where=(hi >= 1.0), color="#2ca02c", alpha=0.10,
                                zorder=1, label="_nolegend_")
                ax.fill_between(xs, hi, 1.0, where=(hi < 1.0), color="#d62728", alpha=0.10,
                                zorder=1, label="_nolegend_")
        # Label the log2 axis in plain multiples (..., 0.5×, 1×, 2×, 4×, ...) rather
        # than 2^n or powers of 10.
        def _mult(v, _):
            if v >= 1:
                return f"{int(round(v))}×" if abs(v - round(v)) < 1e-6 else f"{v:g}×"
            return f"{v:g}×"
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(_mult))
    else:
        # No baseline for this cell -> absolute GiB/s (log y -- spans orders of magnitude).
        ax.set_yscale("log")
        ax.set_ylabel("GiB/s (log)")
        drawn = [a for a in ALGO_ORDER if a in series and len(series[a][0])]
        for i, algo in enumerate(drawn):
            color, marker, label, ls, lw = ALGO_STYLE[algo]
            xb, yv = series[algo]
            any_pts = True
            is_ref = algo in ("default", "main")
            ax.plot(xb, yv, color=color, marker=marker, markersize=10 if is_ref else 6,
                    lw=lw, linestyle=ls, alpha=1.0 if is_ref else 0.8,
                    label=label, zorder=(20 if is_ref else 3 + i))

    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.set_xticks(sorted({int(b) for s in series.values() for b in s[0]}))
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_bins(int(round(v)))))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
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
            xb = [p[0] for p in pts]; yv = [p[1] for p in pts]
            all_bins.update(xb)
            ax.plot(xb, yv, marker="o", ms=5, lw=1.8,
                    color=cmap(0.1 + 0.8 * i / max(1, len(elements_list) - 1)),
                    label=fmt_elements(elements))
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if all_bins:
        ax.set_xticks(sorted(all_bins))
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_bins(int(round(v)))))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    if any_pts:
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7, title="# elements", title_fontsize=7)
    else:
        ax.text(0.5, 0.5, "no hit-rate data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
    return any_pts


def render_one(binary_label, per_algo_cells, sample, shape, elements_list, algos, outpath, hr_for_binary=None):
    """One PNG: characterization top row + per-element-count perf grid, and (when
    hit-rate data is supplied) a final row with cuckoo and single-probe cache
    hit-rate vs #bins (one series per #elements)."""
    bins, counts, char_bins = C.char_input(shape)
    hr_for_binary = hr_for_binary or {}
    has_hr = bool(hr_for_binary.get("direct_cuckoo") or hr_for_binary.get("direct_single_probe"))

    ncols = 3
    nperf = len(elements_list)
    perf_rows = (nperf + ncols - 1) // ncols
    nrows = 1 + perf_rows + (1 if has_hr else 0)

    fig = plt.figure(figsize=(5.4 * ncols, 4.1 * nrows))
    gs = fig.add_gridspec(nrows, ncols)

    transform, channels = BINARY_META[binary_label]
    blurb = C.SHAPE_BLURB.get(shape, "")
    head = f"{transform.upper()} · {channels}-channel · {sample}  —  InputShape: {shape}"
    if blurb:  # only add the parenthetical when a description exists (no empty "()")
        head += f"  ({blurb})"
    head = "\n".join(textwrap.wrap(head, width=120))
    hr_note = "   bottom row: SMEM-cache hit rate vs #bins (series = #elements)" if has_hr else ""
    fig.suptitle(
        f"{head}\n"
        f"top: input characterization (N={C.fmt_int(C.CHAR_N)}, bins={C.fmt_bins(char_bins)})   "
        f"middle: speedup vs baseline (×, log2) vs #bins per input size — one line per algorithm, "
        f"baseline (upstream main) = 1×; default points tagged with the selected algorithm{hr_note}",
        fontsize=12,
    )

    C.draw_distribution(fig.add_subplot(gs[0, 0]), counts, char_bins)
    C.draw_sequence(fig.add_subplot(gs[0, 1]), bins, char_bins, shape=shape)
    legend_ax = fig.add_subplot(gs[0, 2])
    legend_ax.axis("off")
    # Prefix each entry with its 3-letter tag so the legend decodes the tags
    # annotated on the `default` series points (e.g. a "DAC" on the black line maps
    # to "DAC — direct-atomic + cuckoo" here). Without this the tags are unexplained.
    def _legend_label(a):
        tag = ALGO_TAG.get(a)
        return f"{tag} — {ALGO_STYLE[a][2]}" if tag else ALGO_STYLE[a][2]

    handles = [
        plt.Line2D([0], [0], color=ALGO_STYLE[a][0], marker=ALGO_STYLE[a][1],
                   linestyle=ALGO_STYLE[a][3], lw=ALGO_STYLE[a][4], label=_legend_label(a))
        for a in algos
    ]
    legend_ax.legend(handles=handles, loc="center", fontsize=10, title="algorithms (tag — name)",
                     title_fontsize=10, frameon=True)

    for i, elements in enumerate(elements_list):
        r, c = 1 + i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        series = perf_series(per_algo_cells, sample, elements, shape, algos)
        tags = selected_algo_tags(per_algo_cells, sample, elements, shape)
        if not draw_perf(ax, series, f"N = {fmt_elements(elements)} elements", default_tags=tags):
            ax.text(0.5, 0.5, "no data\n(all cells skipped)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

    if has_hr:
        hr_row = 1 + perf_rows
        draw_hitrate(fig.add_subplot(gs[hr_row, 0]), hr_for_binary.get("direct_cuckoo", {}),
                     shape, elements_list, "cuckoo cache — hit rate vs #bins")
        draw_hitrate(fig.add_subplot(gs[hr_row, 1]), hr_for_binary.get("direct_single_probe", {}),
                     shape, elements_list, "single-probe cache — hit rate vs #bins")
        # third column of the hit-rate row: short explainer
        ax = fig.add_subplot(gs[hr_row, 2]); ax.axis("off")
        ax.text(0.5, 0.5, "hit = contribution absorbed in the\nSMEM cache (block-scope add)\n"
                          "miss = spilled to a GMEM atomic\n\n(hit rate is sample-type\nindependent; measured on a\n"
                          "separate instrumented build)",
                ha="center", va="center", fontsize=9, color="#333")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="sweep_results_6shape.json",
                    help="per-cell perf sweep JSON (binary -> algo -> 'SampleT|Elements|Bins|InputShape')")
    ap.add_argument("--hitrate", default="hitrate_results.json",
                    help="per-cell hit-rate sweep JSON (binary -> algo -> 'Bins|Elements|InputShape' -> {rate}); "
                         "optional, adds two cache-hit-rate panels per image when present")
    ap.add_argument("--outdir", default="algo_perf_figs", help="output directory for the per-shape PNGs")
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
        # `smem_privatized` is synthesized from `default` in perf_series (not a data
        # key), so include it whenever `default` is present.
        algos = [a for a in ALGO_ORDER if a in per_algo_cells or a == "smem_privatized"]

        # `_selected` is a meta map (selector's pick per cell), not an algorithm
        # series -- exclude it from sample/element/shape discovery and from plotting.
        data_cells = {a: c for a, c in per_algo_cells.items() if a != "_selected"}
        samples, elements = set(), set()
        for cells in data_cells.values():
            for key in cells:
                s, e, _, _ = key.split("|")
                samples.add(s)
                elements.add(int(e))
        elements = sorted(elements)

        for sample in sorted(samples):
            folder = os.path.join(args.outdir, f"{transform}_{channels}_{sample}")
            os.makedirs(folder, exist_ok=True)
            # Render every shape present in the data for this binary.
            shapes = sorted({k.split("|")[3] for cells in data_cells.values() for k in cells},
                            key=lambda s: (C.SHAPES.index(s) if s in C.SHAPES else 999, s))
            hr_for_binary = hitrate.get(binary_label, {})
            for shape in shapes:
                outpath = os.path.join(folder, f"{shape.replace(':', '_')}.png")
                render_one(binary_label, per_algo_cells, sample, shape, elements, algos, outpath, hr_for_binary)
                written.append(outpath)

    print(f"Wrote {len(written)} images under {args.outdir}")


if __name__ == "__main__":
    main()
