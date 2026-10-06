#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Performance comparison of the high-bin histogram algorithm candidates.

Consumes a per-cell sweep JSON (each high-bin algorithm forced across
SampleT x Elements x Bins x InputShape, per benchmark binary) and the
source-of-truth input generators in `histogram_input_design.py`.

Layout (one image per InputShape, inside a folder per transform/channels/type):
  algo_perf_figs/<even|range>_<single|multi>_<I32|F64>/<input_shape>.png
By default, performance graphs fill the left side while the input characterization,
hit-rate graphs, and legends form a compact right sidebar. `--layout tall` retains
the former top/middle/bottom arrangement. Each performance graph uses X = #bins and
Y = speedup over upstream main, with one measured connect-the-dots series per
algorithm. `default` (the shipping selector's pick, thick black) and `main`
(upstream main's default dispatch, dash-dot red) are the two reference series.

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
import math
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
    "gmem_privatized_nocache": (
        "#9467bd",
        "^",
        "GMEM-privatized + no cache + coalesce on spill",
        "-",
        1.1,
    ),
    "gmem_privatized_agent": (
        "#bcbd22",
        "<",
        "AgentHistogram GMEM-privatized + no cache",
        "--",
        1.1,
    ),
    "hybrid": ("#e377c2", "P", "hybrid (SMEM+GMEM single-pass)", "--", 1.1),
    "gmem_privatized_cuckoo": ("#1f77b4", "o", "gmem-priv + cuckoo", "--", 1.1),
    "gmem_privatized_single_probe": (
        "#17becf",
        "s",
        "GMEM-privatized + single-probe cache + no coalesce",
        ":",
        1.1,
    ),
    "gmem_privatized_nocache_direct_spill": (
        "#7b4ab5",
        "v",
        "GMEM-privatized + no cache + no coalesce",
        ":",
        1.2,
    ),
    "gmem_privatized_single_probe_coalesced_spill": (
        "#007f7f",
        "D",
        "GMEM-privatized + single-probe cache + coalesce on spill",
        "--",
        1.2,
    ),
    "gmem_privatized_single_probe_rle_spill": (
        "#0066cc",
        "P",
        "GMEM-privatized + single-probe cache + RLE on spill",
        "-.",
        1.2,
    ),
    "gmem_privatized_nocache_rle_spill": (
        "#d95f02",
        "X",
        "GMEM-privatized + no cache + RLE on spill",
        "-.",
        1.2,
    ),
    "direct_cuckoo": (
        "#2ca02c",
        "D",
        "direct-atomic + cuckoo SMEM cache",
        "--",
        1.1,
    ),
    "direct_single_probe": ("#8c8c00", "P", "direct-atomic + single-probe", ":", 1.1),
    "direct_nocache": ("#ff7f0e", "v", "direct-atomic + no cache", "-", 1.1),
    # Privatized-SMEM kernel. The CUB_HISTO_FORCE_ALGO hook does not force it (and
    # gates forcing to the high-bin tier), but it IS what the selector runs at bins
    # <= SMEM_PRIVATIZED_MAX_BINS -- so the plotter synthesizes this series from the `default`
    # values over that range (see perf_series). Drawn so the low-bin region is a
    # named algorithm, not two unlabeled reference lines.
    "smem_privatized": ("#8c564b", "h", "smem privatized", "-", 1.3),
    # Low-bin static-vs-dynamic SMEM comparison (histogram_algo_sweep.py forces
    # smem_static / smem_dynamic at <=256 bins): the two privatized-SMEM kernel
    # instantiations measured head-to-head.
    "smem_static": ("#8c564b", "h", "smem privatized (static <=256)", "-", 1.6),
    "smem_dynamic": ("#1f77b4", "D", "smem privatized (dynamic)", "--", 1.3),
    # No-warp-coalesce variants (histogram_algo_sweep.py forces these against the
    # .nocoal binaries). Same base color as the coalesce-on kernel, dotted + open
    # marker to read as "the same algorithm with coalescing disabled".
    "gmem_privatized_nocache__nocoal": (
        "#9467bd",
        "1",
        "gmem-priv gather + no coalesce",
        ":",
        1.3,
    ),
    "direct_nocache__nocoal": (
        "#ff7f0e",
        "2",
        "direct-atomic + no cache + no coalesce",
        ":",
        1.3,
    ),
    "direct_cuckoo__nocoal": (
        "#2ca02c",
        "3",
        "direct-atomic + cuckoo + no coalesce",
        ":",
        1.3,
    ),
    "direct_single_probe__nocoal": (
        "#8c8c00",
        "4",
        "direct-atomic + single-probe + no coalesce",
        ":",
        1.3,
    ),
}
# 3-letter tag per algorithm, used to label each point of the `default` series with
# the algorithm the selector actually picked there. `hybrid` is the smem_split>0
# member of the gmem_privatized_nocache kernel; the dispatch launch tag reports it as
# `gmem_privatized_nocache:hybrid`, which the sweep stores verbatim, so map both the
# member-suffixed name and the bare `hybrid` here.
ALGO_TAG = {
    "smem_privatized": "SMP",
    "gmem_privatized_nocache": "GPN",
    "gmem_privatized_agent": "GPA",
    "gmem_privatized_nocache:hybrid": "HYB",
    "hybrid": "HYB",
    "gmem_privatized_cuckoo": "GPC",
    "gmem_privatized_single_probe": "GPS",
    "gmem_privatized_nocache_direct_spill": "GND",
    "gmem_privatized_single_probe_coalesced_spill": "GSC",
    "gmem_privatized_single_probe_rle_spill": "GSR",
    "gmem_privatized_nocache_rle_spill": "GNR",
    "direct_cuckoo": "DAC",
    "direct_single_probe": "DAS",
    "direct_nocache": "DAN",
    "main": "BAS",  # baseline (upstream main); 3 letters like the rest, ties to "speedup vs baseline"
    "default": "DEF",
    "smem_static": "SST",
    "smem_dynamic": "SDY",
    # no-warp-coalesce variants (match histogram_wins_table.py tags)
    "gmem_privatized_nocache__nocoal": "GN0",
    "direct_nocache__nocoal": "DN0",
    "direct_cuckoo__nocoal": "DC0",
    "direct_single_probe__nocoal": "DS0",
    # The `_selected` map stores the raw CUB_HISTO_LOG_LAUNCH tag, which carries a
    # :static / :dynamic suffix for the privatized-SMEM kernel (the dispatch emits
    # `smem_privatized:dynamic` etc.). Map those suffixed forms too, or the default
    # series gets NO selected-algorithm label at the low-bin tier where the selector
    # picks privatized SMEM (only the high-bin hybrid/direct tags, which have no
    # suffix collision, would otherwise be annotated). SST/SDY tell which kernel ran.
    "smem_privatized:static": "SST",
    "smem_privatized:dynamic": "SDY",
}

# Canonical style/name used to decode annotations on the black `default` line.
# A selected-vs-main-only sweep has no forced-algorithm series, so without these
# proxy entries its SST / SDY / DAS point labels would not appear in the legend.
SELECTED_TAG_ALGO = {
    "SST": "smem_static",
    "SDY": "smem_dynamic",
    "SMP": "smem_privatized",
    "HYB": "hybrid",
    "GPN": "gmem_privatized_nocache",
    "GPA": "gmem_privatized_agent",
    "GPC": "gmem_privatized_cuckoo",
    "GPS": "gmem_privatized_single_probe",
    "GND": "gmem_privatized_nocache_direct_spill",
    "GSC": "gmem_privatized_single_probe_coalesced_spill",
    "GSR": "gmem_privatized_single_probe_rle_spill",
    "GNR": "gmem_privatized_nocache_rle_spill",
    "DAC": "direct_cuckoo",
    "DAS": "direct_single_probe",
    "DAN": "direct_nocache",
}


def algorithm_legend_handles(drawn_algos, selected_tags):
    """Legend handles in canonical algorithm order, with references last.

    A selected algorithm may not have a forced performance series in the input JSON
    (DAS in the July-10 run). Insert that proxy at its normal algorithm-family position
    rather than appending it after BAS/DEF. BAS and DEF are always the final entries.
    """
    handles = []
    represented_tags = set()
    selected_algos = {
        algo for tag, algo in SELECTED_TAG_ALGO.items() if tag in selected_tags
    }

    for algo in (a for a in ALGO_ORDER if a not in ("main", "default")):
        tag = ALGO_TAG.get(algo)
        if algo in drawn_algos:
            color, marker, name, linestyle, linewidth = ALGO_STYLE[algo]
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    lw=linewidth,
                    label=f"{tag} — {name}" if tag else name,
                )
            )
            if tag:
                represented_tags.add(tag)
            continue
        if algo not in selected_algos or not tag or tag in represented_tags:
            continue
        color, marker, name, _linestyle, _linewidth = ALGO_STYLE[algo]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle="none",
                markersize=7,
                label=f"{tag} — selected default ran {name}",
            )
        )
        represented_tags.add(tag)

    # Reference entries are deliberately last: BAS, then DEF.
    for algo in ("main", "default"):
        if algo not in drawn_algos:
            continue
        color, marker, name, linestyle, linewidth = ALGO_STYLE[algo]
        label = (
            "BAS — baseline (upstream main), 1×" if algo == "main" else f"DEF — {name}"
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=linestyle,
                lw=linewidth,
                label=label,
            )
        )
    return handles


def row_major_legend_handles(handles, ncols):
    """Reorder handles so matplotlib's column-major legend reads row-major."""
    if ncols <= 1 or len(handles) <= ncols:
        return handles
    nrows = math.ceil(len(handles) / ncols)
    return [
        handles[index]
        for column in range(ncols)
        for row in range(nrows)
        for index in [row * ncols + column]
        if index < len(handles)
    ]


def grouped_legend_handles(handles, ncols):
    """Put the final BAS/DEF reference pair together on the last legend row."""
    if ncols <= 1 or len(handles) < 2:
        return handles
    reference_count = 0
    for handle in reversed(handles):
        if handle.get_label().startswith(("BAS —", "DEF —")):
            reference_count += 1
        else:
            break
    if reference_count == 0:
        return row_major_legend_handles(handles, ncols)
    candidates, references = handles[:-reference_count], handles[-reference_count:]
    padding = (-len(candidates)) % ncols
    spacers = [
        plt.Line2D([], [], linestyle="none", alpha=0.0, label="")
        for _ in range(padding)
    ]
    return row_major_legend_handles(candidates + spacers + references, ncols)


# Draw order: reference series last (on top). gmem/direct candidates first.
ALGO_ORDER = [
    "smem_privatized",
    "smem_static",
    "smem_dynamic",
    "gmem_privatized_agent",
    "hybrid",
    "gmem_privatized_cuckoo",
    "gmem_privatized_single_probe",
    "gmem_privatized_nocache_direct_spill",
    "gmem_privatized_single_probe_coalesced_spill",
    "gmem_privatized_nocache",
    "gmem_privatized_single_probe_rle_spill",
    "gmem_privatized_nocache_rle_spill",
    "gmem_privatized_nocache__nocoal",
    "direct_cuckoo",
    "direct_single_probe",
    "direct_cuckoo__nocoal",
    "direct_single_probe__nocoal",
    "direct_nocache",
    "direct_nocache__nocoal",
    "main",
    "default",
]

# Cache implementation represented by each hit-rate series. A title names every
# algorithm that shares the cache design, then identifies the specific measured
# series on its second line (e.g. DAS shares GPS's cache but was not measured here).
HITRATE_CACHE_FAMILY = {
    "direct_cuckoo": "cuckoo cache",
    "gmem_privatized_cuckoo": "cuckoo cache",
    "direct_single_probe": "single-probe cache",
    "gmem_privatized_single_probe": "single-probe cache",
    "gmem_privatized_single_probe_coalesced_spill": "single-probe cache",
    "gmem_privatized_single_probe_rle_spill": "single-probe cache",
}
HITRATE_ALGO_ORDER = [
    "direct_cuckoo",
    "gmem_privatized_cuckoo",
    "direct_single_probe",
    "gmem_privatized_single_probe",
    "gmem_privatized_single_probe_coalesced_spill",
    "gmem_privatized_single_probe_rle_spill",
]


def hitrate_cache_title(algo, sample, available_algos=None):
    """Cache-type title with algorithm tags that use that cache.

    By default this preserves the catalog-style label listing every known user of
    the cache family. A focused run can pass its measured hit-rate algorithms so
    the title does not imply that unmeasured algorithms contributed data.
    """
    family = HITRATE_CACHE_FAMILY[algo]
    visible_algos = (
        set(HITRATE_ALGO_ORDER) if available_algos is None else set(available_algos)
    )
    tags = [
        ALGO_TAG[other]
        for other in HITRATE_ALGO_ORDER
        if HITRATE_CACHE_FAMILY[other] == family and other in visible_algos
    ]
    return (
        f"{family} ({', '.join(tags)})\n{ALGO_TAG[algo]} · {sample} hit rate vs #bins"
    )


# Plot the whole swept bin range. The force harness overrides both dispatch
# gates (direct-atomic bin threshold and the hybrid kSplitBin guard) and every
# forced run is launch-validated, so each algorithm -- including hybrid -- is
# genuinely measured down to the smallest swept bin count.
MIN_PLOT_BINS = 0

# LEGACY-ONLY fallback bin cap for the smem_privatized region, used only when a sweep
# JSON predates the `_selected` ground-truth map (both the synthesized smem_privatized
# series and the legacy tag inference prefer `_selected` when present). The real on-chip
# cap is now a per-arch BYTE budget derived at runtime (dispatch_histogram.cuh:
# max_dynamic_smem_bins(counter_bytes, channels, device_optin)), which on B200 admits up to
# ~57344 single-channel bins -- NOT a fixed 16384 -- so this constant is no longer the
# source of truth, just a best-effort guess for pre-`_selected` data.
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


def shape_to_slug(shape: str) -> str:
    """Filesystem-safe, lexically-sortable slug for an InputShape axis string.

    The shape NAME (first ':'-token) is kept verbatim; each subsequent knob
    token is normalized so filenames sort naturally under a plain lexical sort
    (e.g. Windows Explorer). A fractional entropy/hit-rate knob in [0, 10) is
    zero-padded to two decimals (`concentrated:0.5` -> `concentrated_0.50`), so
    that 0.00 < 0.25 < 0.50 < 0.75 < 1.00. Integer-valued tokens (the large
    sawtooth period/stride/scatter params) pass through unchanged rather than
    becoming an absurd `8192.00`. Tokens are joined with '_' and ':' never
    survives, so the result contains no path-illegal characters.
    """
    tokens = shape.split(":")
    out = [tokens[0]]
    for tok in tokens[1:]:
        if "." in tok:
            try:
                val = float(tok)
            except ValueError:
                out.append(tok)  # not a number -- pass through
            else:
                out.append(f"{val:.2f}" if 0.0 <= val < 10.0 else tok)
        else:
            out.append(tok)  # integer-valued (or non-numeric) token, kept as-is
    return "_".join(out)


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
    forced = [
        "gmem_privatized_nocache",
        "gmem_privatized_cuckoo",
        "gmem_privatized_single_probe",
        "direct_cuckoo",
        "direct_single_probe",
        "direct_nocache",
    ]
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
    # Synthesize the `smem_privatized` series: it is never measured under that name
    # (the force hook neither accepts nor needs it), but it IS exactly what the
    # selector `default` runs wherever it chose smem_privatized -- so derive it from the
    # default points over precisely those bins. Makes the low-bin region a named
    # algorithm instead of an unexplained `default`/`main`-only stretch. (Only added
    # when not already an explicit series, so a future real measurement would win.)
    #
    # Ground truth: the `_selected` map records the algorithm the selector actually
    # launched per cell (from the dispatch launch tag). Use it to pick exactly the bins
    # where default == smem_privatized -- robust to the on-chip cap being a per-arch
    # byte budget (no hardcoded bin threshold). For legacy JSON without `_selected`,
    # fall back to the static SMEM_PRIVATIZED_MAX_BINS cap.
    if "smem_privatized" in algos and "smem_privatized" not in out and "default" in out:
        db, dv = out["default"]
        selected = per_algo_cells.get("_selected", {})
        if selected:
            smem_bins = {
                int(b)
                for key, ran in selected.items()
                for s, e, b, sh in [key.split("|")]
                if s == sample
                and int(e) == elements
                and sh == shape
                and ran == "smem_privatized"
            }
            mask = [j for j, b in enumerate(db) if int(b) in smem_bins]
        else:
            mask = [j for j, b in enumerate(db) if int(b) <= SMEM_PRIVATIZED_MAX_BINS]
        if mask:
            out["smem_privatized"] = (
                np.array([db[j] for j in mask]),
                np.array([dv[j] for j in mask]),
            )
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
    any_pts = False

    # CATEGORICAL x-axis: plot bins at evenly-spaced integer positions (0,1,2,...) rather
    # than at their log2 value, so adjacent tiers (e.g. 49152/57344/65536) are not squished
    # together. `xpos` maps a bin count to its slot index over the union of bins present.
    all_bins_sorted = sorted({int(b) for s in series.values() for b in s[0]})
    xpos = {b: i for i, b in enumerate(all_bins_sorted)}

    # Baseline = upstream main. Map bin -> main GiB/s for this (sample, elements, shape).
    baseline = None
    if "main" in series and len(series["main"][0]):
        mb, mv = series["main"]
        baseline = {int(x): y for x, y in zip(mb, mv) if y > 0}

    if baseline:
        # The y-scale (log2 vs linear) is chosen AFTER plotting, from the max speedup in
        # this panel (see below): log2 only when something exceeds 2x (so a wide dynamic
        # range stays legible), otherwise a plain linear scale.
        max_sp = 0.0
        # Every algorithm EXCEPT main, divided by main at the shared bin points.
        drawn = [
            a for a in ALGO_ORDER if a != "main" and a in series and len(series[a][0])
        ]
        for i, algo in enumerate(drawn):
            color, marker, label, ls, lw = ALGO_STYLE[algo]
            xb, yv = series[algo]
            pts = [
                (int(x), y / baseline[int(x)])
                for x, y in zip(xb, yv)
                if int(x) in baseline
            ]
            if not pts:
                continue
            any_pts = True
            xs = np.array([xpos[p[0]] for p in pts], dtype=float)
            sp = np.array([p[1] for p in pts])
            max_sp = max(max_sp, float(sp.max()))
            is_ref = algo == "default"
            # Lines drawn first (lower zorder); markers all sit ON TOP (zorder 30+)
            # at the TRUE x, semi-transparent so overlapping points blend visibly
            # (two coincident markers read darker / show both colors) rather than one
            # opaque marker hiding another. Distinct marker shapes + dash patterns
            # still tell coincident series apart; no x-dodge (true data positions).
            # The `default` series is also semi-transparent (both line and marker) so
            # the candidate series it rides on top of remain visible underneath.
            line_alpha = 0.5 if is_ref else 0.85
            mark_alpha = 0.5 if is_ref else 0.6
            ax.plot(
                xs,
                sp,
                color=color,
                lw=lw,
                linestyle=ls,
                marker="",
                alpha=line_alpha,
                label=label,
                zorder=(20 if is_ref else 3 + i),
            )
            ax.plot(
                xs,
                sp,
                color=color,
                marker=marker,
                markersize=8 if is_ref else 5.5,
                linestyle="none",
                alpha=mark_alpha,
                zorder=(32 if is_ref else 30 + i),
            )
            # Annotate each default point with the selected algorithm's 3-letter tag.
            # `pts` carries the true bin counts; x position is the categorical slot.
            if is_ref and default_tags:
                for b, y in pts:
                    tag = default_tags.get(int(b))
                    if tag:
                        ax.annotate(
                            tag,
                            (xpos[int(b)], y),
                            textcoords="offset points",
                            xytext=(0, 7),
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            fontweight="bold",
                            color="#000000",
                            zorder=40,
                        )
        # Baseline reference: main is 1x by definition. Draw it as the styled main line.
        b_color, _, _, b_ls, b_lw = ALGO_STYLE["main"]
        ax.axhline(
            1.0,
            color=b_color,
            linestyle=b_ls,
            lw=b_lw,
            alpha=1.0,
            zorder=19,
            label="baseline (upstream main), 1×",
        )
        # Shade the shipping default's gain over the baseline (between 1x and default).
        if "default" in series and len(series["default"][0]):
            db, dv = series["default"]
            dpts = sorted(
                (int(x), y / baseline[int(x)])
                for x, y in zip(db, dv)
                if int(x) in baseline
            )
            if dpts:
                xs = np.array([xpos[p[0]] for p in dpts])
                hi = np.array([p[1] for p in dpts])
                ax.fill_between(
                    xs,
                    1.0,
                    hi,
                    where=(hi >= 1.0),
                    color="#2ca02c",
                    alpha=0.10,
                    zorder=1,
                    label="_nolegend_",
                )
                ax.fill_between(
                    xs,
                    hi,
                    1.0,
                    where=(hi < 1.0),
                    color="#d62728",
                    alpha=0.10,
                    zorder=1,
                    label="_nolegend_",
                )
        # Y-scale chosen from the data: log2 ONLY when some series exceeds 2x (a wide
        # dynamic range that a linear axis would crush against the top); otherwise a
        # plain linear scale (more readable when everything sits near 1x). Either way
        # ticks are labelled in plain multiples (..., 0.5x, 1x, 2x, 4x, ...).
        if max_sp > 2.0:
            ax.set_yscale("log", base=2)
            ax.set_ylabel("speedup vs baseline (×, log2)")
        else:
            ax.set_yscale("linear")
            ax.set_ylabel("speedup vs baseline (×)")

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
            ax.plot(
                [xpos[int(x)] for x in xb],
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

    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    # Categorical ticks: one slot per bin, evenly spaced, labelled with the bin count.
    ax.set_xticks(range(len(all_bins_sorted)))
    ax.set_xticklabels([fmt_bins(b) for b in all_bins_sorted])
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    return any_pts


def _hitrate_key_parts(key):
    """Return (sample, bins, elements, shape); accept legacy I32-only keys."""
    parts = key.split("|")
    if len(parts) == 4:
        return parts
    if len(parts) == 3:
        bins, elements, shape = parts
        return "I32", bins, elements, shape
    raise ValueError(f"invalid hit-rate key: {key!r}")


def draw_hitrate(ax, hr_algo_cells, sample, shape, elements_list, title):
    """Cache hit rate (%) vs #bins, one series per #elements. hr_algo_cells is
    keyed 'SampleT|Bins|Elements|InputShape' -> {rate,hits,misses}. Legacy
    three-field keys are treated as I32-only."""
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("# bins")
    ax.set_ylabel("cache hit rate (%)")
    any_pts = False
    cmap = plt.cm.viridis
    # CATEGORICAL x-axis (evenly spaced), matching draw_perf: map each bin to a slot index.
    matching = [
        (key, rec, _hitrate_key_parts(key))
        for key, rec in hr_algo_cells.items()
        if _hitrate_key_parts(key)[0] == sample
    ]
    all_bins_sorted = sorted({int(parts[1]) for _key, _rec, parts in matching})
    xpos = {b: i for i, b in enumerate(all_bins_sorted)}
    for i, elements in enumerate(elements_list):
        pts = []
        for _key, rec, parts in matching:
            _sample, b, e, sh = parts
            if int(e) == elements and sh == shape:
                pts.append((int(b), rec["rate"] * 100.0))
        pts.sort()
        if pts:
            any_pts = True
            xb = [xpos[p[0]] for p in pts]
            yv = [p[1] for p in pts]
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
    if all_bins_sorted:
        ax.set_xticks(range(len(all_bins_sorted)))
        ax.set_xticklabels([fmt_bins(b) for b in all_bins_sorted])
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    if any_pts:
        ax.set_ylim(-2, 102)
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


def geomean_perf_series(per_algo_cells, sample, elements, algos, shapes):
    """Like perf_series, but each (algo, bins) point is the GEOMEAN of that algorithm's
    GiB/s across all `shapes` at that (sample, elements, bins). Only bins where the algo
    has >=1 positive shape sample contribute; the geomean is over exactly the shapes
    present (so a forced algo that is dropped on some shapes still gets a fair mean over
    the rest). Returns {algo: (bins[], geomean_gibs[])}, with `smem_privatized` synthesized
    from `default` over the bins the selector ran it (same rule as perf_series)."""
    shape_set = set(shapes)
    out = {}
    for algo in algos:
        cells = per_algo_cells.get(algo, {})
        bybin = {}
        for key, gibs in cells.items():
            s, e, b, sh = key.split("|")
            if (
                s == sample
                and int(e) == elements
                and sh in shape_set
                and int(b) >= MIN_PLOT_BINS
                and gibs > 0
            ):
                bybin.setdefault(int(b), []).append(gibs)
        pts = sorted(
            (b, math.exp(sum(map(math.log, v)) / len(v))) for b, v in bybin.items()
        )
        if pts:
            out[algo] = (
                np.array([p[0] for p in pts], dtype=float),
                np.array([p[1] for p in pts]),
            )
    # Synthesize smem_privatized from the geomean default over the bins where the
    # selector ran smem_privatized (the pick is shape-independent, so the bin set is the
    # union across shapes), mirroring perf_series.
    if "smem_privatized" in algos and "smem_privatized" not in out and "default" in out:
        db, dv = out["default"]
        selected = per_algo_cells.get("_selected", {})
        smem_bins = set()
        if selected:
            for key, ran in selected.items():
                s, e, b, sh = key.split("|")
                if (
                    s == sample
                    and int(e) == elements
                    and sh in shape_set
                    and ran == "smem_privatized"
                ):
                    smem_bins.add(int(b))
        else:
            smem_bins = {int(b) for b in db if int(b) <= SMEM_PRIVATIZED_MAX_BINS}
        pts = [(int(b), v) for b, v in zip(db, dv) if int(b) in smem_bins]
        if pts:
            out["smem_privatized"] = (
                np.array([p[0] for p in pts], dtype=float),
                np.array([p[1] for p in pts]),
            )
    return out


def geomean_selected_tags(per_algo_cells, sample, elements, shapes):
    """{bins: tag} for the geomean default series. The selector is shape-blind, so its
    pick is (almost always) constant across shapes at a given (sample, elements, bins);
    take the pick of the first shape that has one. Falls back to whatever tag exists."""
    selected = per_algo_cells.get("_selected", {})
    tags = {}
    for key, ran in selected.items():
        s, e, b, sh = key.split("|")
        if s == sample and int(e) == elements and sh in set(shapes) and ran in ALGO_TAG:
            tags.setdefault(
                int(b), ALGO_TAG[ran]
            )  # first shape wins; selector is shape-blind
    return tags


def render_geomean(
    binary_label,
    per_algo_cells,
    sample,
    elements_list,
    algos,
    shapes,
    outpath,
    layout="wide",
):
    """One PNG: speedup-vs-#bins for each algorithm, GEOMEAN over all input shapes, one
    panel per element count. Same layout/style as render_one's middle grid, but WITHOUT
    the two input-characterization panels (a geomean is shape-agnostic) and without the
    per-shape hit-rate row. The legend takes the slot the characterization row vacated."""
    nperf = len(elements_list)
    ncols = 4 if layout == "wide" and nperf > 6 else 3
    perf_rows = (nperf + ncols - 1) // ncols
    free_slots = perf_rows * ncols - nperf
    # When the perf grid is exactly full (no free cell for the legend, e.g. 6 panels in a
    # 2x3 grid), add a dedicated short bottom ROW for the legend so it never overlaps the
    # panels' x-axis labels. A small height_ratio keeps that row compact.
    needs_legend_row = layout == "wide" or free_slots == 0
    nrows = perf_rows + (1 if needs_legend_row else 0)

    # In the wide layout, match the overall width used by the per-shape figures.
    # Geomean omits their characterization sidebar, but collapsing that space made
    # all_geomean.png roughly 40% narrower than every neighboring image and caused
    # an unpleasant display-size jump when browsing the run. Give that space to the
    # performance panels instead.
    figure_width = 5.0 * ncols + 11.5 if layout == "wide" else 5.4 * ncols
    fig = plt.figure(
        figsize=(
            figure_width,
            4.1 * perf_rows + (0.9 if needs_legend_row else 0),
        )
    )
    height_ratios = [4.1] * perf_rows + ([0.9] if needs_legend_row else [])
    gs = fig.add_gridspec(nrows, ncols, height_ratios=height_ratios)

    transform, channels = BINARY_META[binary_label]
    head = f"{transform.upper()} · {channels}-channel · {sample}  —  GEOMEAN over {len(shapes)} input shapes"
    if "main" in per_algo_cells:
        subtitle = (
            "speedup vs baseline (×, log2) vs #bins per input size — geomean of GiB/s across all "
            "shapes; one line per algorithm, baseline (upstream main) = 1×; default points tagged "
            "with the selected algorithm"
        )
    else:
        subtitle = (
            "absolute GiB/s vs #bins per input size — geomean across all input shapes"
        )
    fig.suptitle(
        head + "\n" + "\n".join(textwrap.wrap(subtitle, width=140)), fontsize=12
    )

    drawn_algos = set()
    selected_point_tags = set()
    for i, elements in enumerate(elements_list):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        series = geomean_perf_series(per_algo_cells, sample, elements, algos, shapes)
        tags = geomean_selected_tags(per_algo_cells, sample, elements, shapes)
        selected_point_tags.update(tags.values())
        # Record which series actually produced points (for an honest legend).
        drawn_algos.update(a for a in series if len(series[a][0]))
        if not draw_perf(
            ax, series, f"N = {fmt_elements(elements)} elements", default_tags=tags
        ):
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

    # Legend, in a free grid cell for the legacy tall layout when possible, otherwise
    # in a compact dedicated row. The shared builder places selected-only proxies in
    # family order and makes BAS/DEF the final entries.
    handles = algorithm_legend_handles(drawn_algos, selected_point_tags)
    if free_slots > 0 and not needs_legend_row:
        # Legend in the first unused panel cell of the last perf row.
        lax = fig.add_subplot(gs[perf_rows - 1, ncols - free_slots])
        lax.axis("off")
        lax.legend(
            handles=handles,
            loc="center",
            fontsize=10,
            title="algorithms (tag — name)",
            title_fontsize=10,
            frameon=True,
        )
    else:
        # Dedicated bottom legend row spanning all columns -- its own reserved space, so
        # it cannot overlap the panels' "# bins" labels above it.
        lax = fig.add_subplot(gs[perf_rows, :])
        lax.axis("off")
        legend_ncols = min(len(handles), 5)
        lax.legend(
            handles=grouped_legend_handles(handles, legend_ncols),
            loc="center",
            ncol=legend_ncols,
            fontsize=9,
            title="algorithms (tag — name)",
            title_fontsize=9,
            frameon=True,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_one(
    binary_label,
    per_algo_cells,
    sample,
    shape,
    elements_list,
    algos,
    outpath,
    hr_for_binary=None,
    layout="wide",
    hitrate_label_scope="family",
):
    """Render one per-shape PNG in the selectable wide or legacy tall layout."""
    bins, counts, char_bins = C.char_input(shape)
    hr_for_binary = hr_for_binary or {}

    def has_sample(cells):
        return any(_hitrate_key_parts(key)[0] == sample for key in cells)

    hitrate_algos = [
        algo for algo in HITRATE_ALGO_ORDER if has_sample(hr_for_binary.get(algo, {}))
    ]
    has_hr = bool(hitrate_algos)

    nperf = len(elements_list)
    if layout == "wide":
        perf_cols = 4 if nperf > 6 else 3
        perf_rows = (nperf + perf_cols - 1) // perf_cols
        figure_height = max(8.8, 4.0 * perf_rows + 0.8)
        fig = plt.figure(figsize=(5.0 * perf_cols + 11.5, figure_height))
        outer = fig.add_gridspec(1, 2, width_ratios=[perf_cols, 2.35], wspace=0.12)
        perf_gs = outer[0, 0].subgridspec(
            perf_rows, perf_cols, hspace=0.38, wspace=0.40
        )
        perf_axes = [
            fig.add_subplot(perf_gs[i // perf_cols, i % perf_cols])
            for i in range(nperf)
        ]

        hr_rows = (len(hitrate_algos) + 1) // 2 if has_hr else 0
        side_rows = 1 + hr_rows + 1  # characterization + hit rates + algorithm legend
        side_gs = outer[0, 1].subgridspec(
            side_rows,
            2,
            height_ratios=[1.0] + [1.0] * hr_rows + [0.72],
            hspace=0.68,
            wspace=0.32,
        )
        distribution_ax = fig.add_subplot(side_gs[0, 0])
        sequence_ax = fig.add_subplot(side_gs[0, 1])
        hitrate_axes = []
        for i, _algo in enumerate(hitrate_algos):
            hitrate_axes.append(fig.add_subplot(side_gs[1 + i // 2, i % 2]))
        for row in range(hr_rows):
            used = min(2, max(0, len(hitrate_algos) - 2 * row))
            for column in range(used, 2):
                fig.add_subplot(side_gs[1 + row, column]).axis("off")
        bottom_gs = side_gs[-1, :].subgridspec(
            1, 2, width_ratios=[1.55, 1.0], wspace=0.15
        )
        legend_ax = fig.add_subplot(bottom_gs[0, 0])
        hitrate_legend_ax = fig.add_subplot(bottom_gs[0, 1])
        hitrate_legend_ax.axis("off")
    else:
        perf_cols = 3
        perf_rows = (nperf + perf_cols - 1) // perf_cols
        hr_rows = (len(hitrate_algos) + 1) // 2 if has_hr else 0
        nrows = 1 + perf_rows + hr_rows
        fig = plt.figure(figsize=(5.4 * perf_cols, 4.1 * nrows))
        gs = fig.add_gridspec(nrows, perf_cols)
        distribution_ax = fig.add_subplot(gs[0, 0])
        sequence_ax = fig.add_subplot(gs[0, 1])
        legend_ax = fig.add_subplot(gs[0, 2])
        perf_axes = [
            fig.add_subplot(gs[1 + i // perf_cols, i % perf_cols]) for i in range(nperf)
        ]
        hitrate_axes = []
        hitrate_legend_ax = None
        for i, _algo in enumerate(hitrate_algos):
            hitrate_axes.append(fig.add_subplot(gs[1 + perf_rows + i // 2, i % 2]))
        for row in range(hr_rows):
            used = min(2, max(0, len(hitrate_algos) - 2 * row))
            for column in range(used, 2):
                fig.add_subplot(gs[1 + perf_rows + row, column]).axis("off")
            side_ax = fig.add_subplot(gs[1 + perf_rows + row, 2])
            side_ax.axis("off")
            if row == 0:
                hitrate_legend_ax = side_ax

    transform, channels = BINARY_META[binary_label]
    blurb = C.SHAPE_BLURB.get(shape, "")
    head = (
        f"{transform.upper()} · {channels}-channel · {sample}  —  InputShape: {shape}"
    )
    if blurb:  # only add the parenthetical when a description exists (no empty "()")
        head += f"  ({blurb})"
    wrap_width = 210 if layout == "wide" else 120
    head = "\n".join(textwrap.wrap(head, width=wrap_width))
    perf_note = (
        "speedup vs baseline vs #bins; baseline (upstream main) = 1×; default points tagged "
        "with the selected algorithm"
        if "main" in per_algo_cells
        else "absolute GiB/s vs #bins"
    )
    if layout == "wide":
        sidebar_note = (
            f"; cache hit rate vs #bins ({sample}, series = #elements)"
            if has_hr
            else ""
        )
        subtitle = (
            f"left: {perf_note}, one panel per input size   "
            f"right: input characterization (N={C.fmt_int(C.CHAR_N)}, "
            f"bins={C.fmt_bins(char_bins)}){sidebar_note}"
        )
    else:
        hr_note = (
            f"   bottom: {sample} cache hit rate vs #bins (series = #elements)"
            if has_hr
            else ""
        )
        subtitle = (
            f"top: input characterization (N={C.fmt_int(C.CHAR_N)}, "
            f"bins={C.fmt_bins(char_bins)})   middle: {perf_note}, one panel per input size"
            f"{hr_note}"
        )
    fig.suptitle(
        head + "\n" + "\n".join(textwrap.wrap(subtitle, width=wrap_width)),
        fontsize=12,
    )

    C.draw_distribution(distribution_ax, counts, char_bins)
    C.draw_sequence(sequence_ax, bins, char_bins, shape=shape)
    if layout == "wide":
        distribution_ax.set_title(
            "value distribution (count vs bin, log y)", fontsize=8
        )
        sequence_ax.set_title("position vs bin index", fontsize=8)

    # Draw the per-element-count perf panels FIRST, recording which series actually
    # produced points, so the legend lists only series present in this figure's data.
    drawn_algos = set()
    selected_point_tags = set()
    for ax, elements in zip(perf_axes, elements_list):
        series = perf_series(per_algo_cells, sample, elements, shape, algos)
        tags = selected_algo_tags(per_algo_cells, sample, elements, shape)
        selected_point_tags.update(tags.values())
        drawn_algos.update(a for a in series if len(series[a][0]))
        if not draw_perf(
            ax, series, f"N = {fmt_elements(elements)} elements", default_tags=tags
        ):
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

    # The shared ordered legend puts selected-only DAS directly after DAC and keeps
    # the two references, BAS and DEF, as the final entries.
    legend_ax.axis("off")
    handles = algorithm_legend_handles(drawn_algos, selected_point_tags)
    legend_ncols = 2 if layout == "wide" else 1
    legend_ax.legend(
        handles=grouped_legend_handles(handles, legend_ncols),
        loc="center",
        ncol=legend_ncols,
        fontsize=8.5 if layout == "wide" else 10,
        title="algorithms (tag — name)",
        title_fontsize=9 if layout == "wide" else 10,
        frameon=True,
    )

    if has_hr:
        for hr_ax, algo in zip(hitrate_axes, hitrate_algos):
            draw_hitrate(
                hr_ax,
                hr_for_binary.get(algo, {}),
                sample,
                shape,
                elements_list,
                hitrate_cache_title(
                    algo,
                    sample,
                    hitrate_algos if hitrate_label_scope == "measured" else None,
                ),
            )
        # ONE shared #elements legend below the context graphs, beside the
        # algorithm-series legend. It is never overlaid on a data panel.
        hr_handles, hr_labels = hitrate_axes[0].get_legend_handles_labels()
        if hr_handles and hitrate_legend_ax is not None:
            hitrate_legend_ax.legend(
                hr_handles,
                hr_labels,
                loc="center",
                fontsize=8,
                title="# elements",
                title_fontsize=8,
                frameon=True,
            )

    if layout == "wide":
        # Nested performance/sidebar GridSpecs already own their internal spacing;
        # tight_layout does not understand that nesting and emits one warning per
        # figure. Set only the outer margins here and leave each subgrid intact.
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.065, top=0.86)
    else:
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
        help="per-cell hit-rate sweep JSON "
        "(binary -> algo -> 'SampleT|Bins|Elements|InputShape' -> {rate}); "
        "optional, adds two cache-hit-rate panels per image when present",
    )
    ap.add_argument(
        "--outdir",
        default="algo_perf_figs",
        help="output directory for the per-shape PNGs",
    )
    ap.add_argument(
        "--layout",
        choices=("wide", "tall"),
        default="wide",
        help="figure arrangement: laptop-friendly wide sidebar (default) or legacy tall stack",
    )
    ap.add_argument(
        "--hitrate-label-scope",
        choices=("family", "measured"),
        default="family",
        help="label hit-rate panels with every cache-family user (default) or only "
        "algorithms measured in the supplied hit-rate JSON",
    )
    args = ap.parse_args()

    # The characterization helper is shared with the standalone catalog. Keep
    # its block-0 grid-stride overlay under the same executable regression check
    # when rendering composite performance figures directly.
    C.validate_block_stride_overlay()

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
        if binary_label.startswith("_"):
            continue  # provenance/meta (e.g. _meta), not a binary
        transform, channels = BINARY_META[binary_label]
        # Plot every algorithm series actually present in the data for this binary
        # (in canonical ALGO_ORDER). `main` appears only for the generator-identical
        # shapes (the sweep driver omits it elsewhere); a forced algo absent at a
        # given (transform, channels) simply has no points and is dropped per panel.
        # `smem_privatized` is synthesized from `default` in perf_series (not a data
        # key), so include it whenever `default` is present -- EXCEPT when the data has
        # explicit `smem_static` / `smem_dynamic` series (the low-bin static-vs-dynamic
        # comparison sweep). There the synthesized SMP line is redundant with the
        # measured SST/SDY series (it just traces one of them), so drop it as clutter.
        has_explicit_smem = ("smem_static" in per_algo_cells) or (
            "smem_dynamic" in per_algo_cells
        )
        algos = [
            a
            for a in ALGO_ORDER
            if a in per_algo_cells or (a == "smem_privatized" and not has_explicit_smem)
        ]

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
            shapes = sorted(
                {k.split("|")[3] for cells in data_cells.values() for k in cells},
                key=lambda s: (C.SHAPES.index(s) if s in C.SHAPES else 999, s),
            )
            hr_for_binary = hitrate.get(binary_label, {})
            for shape in shapes:
                outpath = os.path.join(folder, f"{shape_to_slug(shape)}.png")
                render_one(
                    binary_label,
                    per_algo_cells,
                    sample,
                    shape,
                    elements,
                    algos,
                    outpath,
                    hr_for_binary,
                    args.layout,
                    args.hitrate_label_scope,
                )
                written.append(outpath)
            # One geomean-over-shapes figure per (binary, sample): the shape-agnostic
            # summary, omitting the two per-shape input-characterization panels and the
            # per-shape hit-rate row.
            geo_path = os.path.join(folder, "all_geomean.png")
            render_geomean(
                binary_label,
                per_algo_cells,
                sample,
                elements,
                algos,
                shapes,
                geo_path,
                args.layout,
            )
            written.append(geo_path)

    print(f"Wrote {len(written)} images under {args.outdir}")


if __name__ == "__main__":
    main()
