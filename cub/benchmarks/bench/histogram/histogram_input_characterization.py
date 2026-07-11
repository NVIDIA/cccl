#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Characterization plots for the CUB histogram benchmark InputShapes.

For every InputShape (see `histogram_inputs.cuh`), draw what the input actually
looks like, straight from the SOURCE OF TRUTH generators in
`histogram_input_design.py` (the bit-exact host mirror of the .cuh). One figure
per shape, three panels each:

  1. value distribution  -- count vs bin index, on a LOG y-axis with a stem per
     occupied bin. The earlier linear count-vs-index plot made these look empty:
     a single hot bin is one sub-pixel-wide spike among thousands of bins, and a
     decaying tail (counts 1..100) is crushed flat under a ~N-tall spike. Log-y
     stems show both the spike(s) AND the floor; the hottest bins are annotated
     with their real index (so e.g. `concentrated:0.0`'s hot bin reads as
     "bin 42", which is `seed % num_bins` -- deliberately scattered off zero --
     not "empty at zero").
  2. rank-frequency       -- sorted count vs rank, log-log. Scale-free: a
     power law is a straight line, while a single hot bin is one
     point above an empty floor, hash_synonym a few high points over a flat
     plateau. This panel reveals the distribution's shape regardless of bin
     count, so it never "looks empty".
  3. position of values   -- bin index vs position in the input sequence
     (subsampled). i.i.d. shapes smear vertically over their occupied bins;
     ORDERING shapes show their structure here (temporal_phases steps,
     stale_resident's uniform visits to its cold working set, the sawtooth's
     ramp-and-reset).

This is the standalone characterization generator AND the home of the shared
draw_* functions reused by `histogram_algo_perf.py` for its top-row context.

Run with a Python that has numpy + matplotlib, e.g.:
  python histogram_input_characterization.py --outdir histogram_input_figs
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import sys
import textwrap

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import histogram_input_design as D  # noqa: E402

# ---------------------------------------------------------------------------
# Shapes + human-readable blurbs. Mirrors the catalog in histogram_inputs.cuh.
# ---------------------------------------------------------------------------
SHAPES = [
    "concentrated:1.0",
    "concentrated:0.75",
    "concentrated:0.5",
    "concentrated:0.25",
    "concentrated:0.0",
    "powerlaw:0.75",
    "powerlaw:0.5",
    "powerlaw:0.25",
    "temporal_phases:0.10",
    "stale_resident:0.5",
    "stale_resident:0.25",
    "hash_synonym",
    "sawtooth",
    "poison",
    "sawtooth:8192:2654435761:1",
]

SHAPE_BLURB = {
    "concentrated:1.0": "uniform endpoint (entropy 1.0): exact equal counts per bin, in random sequence order "
    "(a Feistel shuffle of the tiling) — uniform counts, randomly distributed",
    "concentrated:0.75": "entropy 0.75: random bin probabilities (softmax over random logits) — MOSTLY uniform, mild variation",
    "concentrated:0.5": "entropy 0.5 (bare `concentrated` default): random bin probabilities dialed to medium entropy",
    "concentrated:0.25": "entropy 0.25: random bin probabilities, now strongly skewed toward a few bins",
    "concentrated:0.0": "single value (entropy 0.0): 100% on one bin, scattered to seed%bins (NOT bin 0)",
    "powerlaw:0.75": "power-law warm set (entropy 0.75): gentle 1/rank^s decay over many bins",
    "powerlaw:0.5": "power-law warm set (entropy 0.5): one dominant bin + a decaying tail",
    "powerlaw:0.25": "power-law warm set (entropy 0.25): steep decay, a few bins dominate",
    "temporal_phases": "eight evenly spaced temporal hot bins with 10% uniform random noise in each phase",
    "temporal_phases:0.10": "eight evenly spaced temporal hot bins with 10% uniform random noise in each phase",
    "stale_resident:0.5": "W=2S cold bins in an upper-bin window vs an S-slot no-eviction cache "
    "(targeting ~50% hits)",
    "stale_resident:0.25": "W=4S cold bins in an upper-bin window vs an S-slot no-eviction cache "
    "(targeting ~25% hits)",
    "hash_synonym": "up to 32 hot bins from one real primary-hash bucket; a short prefix primes their "
    "secondary slots so both one- and two-probe caches reject the collisions",
    "sawtooth": "bin(i)=i%period: a monotonic ramp that resets periodically (sequential locality, bounded working set)",
    "poison": "two blocker bins claim the cache slots probed by one poison bin, then the poison bin dominates and misses",
    "sawtooth:8192:2654435761:1": "8192 upper-window bins visited in a coprime strided cycle "
    "without a wrapped low-bin tail",
}

# Characterization sample size. These are illustrative inputs, not the multi-GB
# benchmark inputs.
CHAR_N = 1_000_000
CHAR_SEQ_SAMPLES = 4000
CHAR_SEED = 42

# Bin count is PER SHAPE, drawn at the shape's natural scale (the original design
# figures did the same). Two reasons it must not be one global value:
#  * the i.i.d. distribution shapes put their hot bin at seed % num_bins (= 42).
#    At 16384 bins that is 0.3% from the left edge and reads as "pinned to 0";
#    at a few hundred bins it sits visibly off zero where it belongs.
#  * the cache-adversarial shapes are defined RELATIVE TO the SMEM cache slot count,
#    so their structure only appears for bins > the working set (hash_synonym needs
#    enough bins per actual cache slot to expose many synonyms; stale_resident needs bins > W so the
#    whole working set fits, e.g. W=4S=32768 for :0.25 at the figure's S=8192).
CHAR_BINS_DEFAULT = 16384
CHAR_BINS_BY_SHAPE = {
    "concentrated": 256,  # hot bin at 42 clearly off zero; floor legible
    "powerlaw": 256,
    "temporal_phases": 256,  # the 8 phase locations are distinct
    "sawtooth": 256,  # several ramp periods visible
    "sawtooth:8192:2654435761:1": 16384,  # show the full upper-window 8192-bin support
    "poison": 32768,  # at least four bins per representative 8192-slot cache entry
    "hash_synonym": 262144,  # ~32 real primary-slot synonyms for the representative S=8192
    "stale_resident": 65536,  # bins > W (W up to 4*8192 for :0.25) so the whole set fits
}


def bins_for_shape(shape: str) -> int:
    """The bin count this shape is characterized at (keyed by name, ignoring knob)."""
    return CHAR_BINS_BY_SHAPE.get(
        shape, CHAR_BINS_BY_SHAPE.get(shape.split(":")[0], CHAR_BINS_DEFAULT)
    )


def fmt_bins(b: int) -> str:
    b = int(b)
    return f"{b // 1024}K" if b >= 1024 and b % 1024 == 0 else str(b)


DIST_COLOR = "#2b8cbe"
RANK_COLOR = "#6a51a3"
SEQ_COLOR = "#cb181d"


def fmt_int(v: int) -> str:
    return f"{int(v):,}"


@functools.lru_cache(maxsize=None)
def _counts_cached(shape: str, n: int, num_bins: int, seed: int):
    """(bins, counts) for a shape. Cached so the perf script's 64 composites
    regenerate each shape's input only once."""
    bins = np.asarray(D.generate_bins(shape, n, num_bins, seed=seed), dtype=np.int64)
    counts = np.bincount(bins, minlength=num_bins).astype(np.int64)
    return bins, counts


def char_input(
    shape: str, n: int = CHAR_N, num_bins: int | None = None, seed: int = CHAR_SEED
):
    """(bins, counts, num_bins) for a shape, at its natural per-shape bin count."""
    if num_bins is None:
        num_bins = bins_for_shape(shape)
    bins, counts = _counts_cached(shape, n, num_bins, seed)
    return bins, counts, num_bins


# ---------------------------------------------------------------------------
# Shared draw_* functions (reused by histogram_algo_perf.py).
# ---------------------------------------------------------------------------


def draw_distribution(ax, counts, num_bins):
    """count vs bin index on a log y-axis, one stem per occupied bin.

    Robust to the spiky, wide distributions here: a lone hot bin shows as a tall
    stem with a marker (never sub-pixel-invisible), and a low floor shows as a
    band well above the axis floor instead of being crushed to zero."""
    nz = np.flatnonzero(counts)
    if nz.size == 0:
        ax.text(
            0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes
        )
        return
    cmax = int(counts[nz].max())
    base = 0.7  # < 1 so a count of 1 still draws a short stem on the log axis
    ax.set_yscale("log")
    # Thin stems when nearly every bin is occupied (uniform / sawtooth),
    # thicker when there are a few discrete spikes.
    dense = nz.size > 2000
    ax.vlines(
        nz, base, counts[nz], color=DIST_COLOR, lw=0.5 if dense else 1.6, alpha=0.9
    )
    ax.scatter(
        nz,
        counts[nz],
        s=8 if dense else 22,
        color=DIST_COLOR,
        zorder=3,
        edgecolors="none",
    )

    ax.set_title(
        f"value distribution (count vs bin index, {fmt_bins(num_bins)} bins, log y)",
        fontsize=10,
    )
    ax.set_xlabel("bin index")
    ax.set_ylabel("count (log)")
    ax.set_xlim(-num_bins * 0.02, num_bins * 1.02)
    ax.set_ylim(base, cmax * 2.5)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.45)


def draw_rankfreq(ax, counts):
    """Sorted count vs rank, log-log. Scale-free view of the distribution shape."""
    c = np.sort(counts[counts > 0])[::-1]
    if c.size == 0:
        ax.text(
            0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes
        )
        return
    ranks = np.arange(1, c.size + 1)
    ax.loglog(ranks, c, marker=".", ms=4, lw=1.1, color=RANK_COLOR)
    ax.set_title(
        f"rank-frequency (sorted count vs rank, log-log) — {c.size} occupied bins",
        fontsize=10,
    )
    ax.set_xlabel("rank (hottest = 1)")
    ax.set_ylabel("count")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)


# Shapes whose interesting sequence structure lives at the WHOLE-sequence scale
# (e.g. the hot bin steps across the full input, or poison transitions from its
# short claim phase to its long miss phase), so the panel must span all N.
# Everything else is shown as a CONTIGUOUS prefix: a periodic shape (sawtooth)
# aliases into garbage if you linspace-subsample N points at a period that divides
# the step, but a contiguous window never aliases and shows the true ramp; the
# stale_resident scan is a uniform draw over its working set, well represented by a
# contiguous window too. For i.i.d. shapes a contiguous window is just as
# representative as a random subsample.
_FULL_RANGE_SEQ_SHAPES = {"hash_synonym", "poison", "temporal_phases"}


def draw_sequence(ax, bins, num_bins, shape=None):
    """bin index vs position in the input sequence."""
    n = len(bins)
    full_range = shape is not None and shape.split(":")[0] in _FULL_RANGE_SEQ_SHAPES
    if full_range:
        idx = (
            np.linspace(0, n - 1, CHAR_SEQ_SAMPLES).astype(np.int64)
            if n > CHAR_SEQ_SAMPLES
            else np.arange(n)
        )
        xlabel = "position in input sequence (full range, subsampled)"
    else:
        w = min(n, CHAR_SEQ_SAMPLES)
        idx = np.arange(w)  # contiguous prefix — never aliases periodic shapes
        xlabel = f"position in input sequence (first {w:,})"
    ax.scatter(idx, bins[idx], s=5, alpha=0.45, color=SEQ_COLOR, edgecolors="none")
    ax.set_title("position of values (bin index vs position in sequence)", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("bin index")
    ax.set_ylim(-num_bins * 0.02, num_bins * 1.02)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ---------------------------------------------------------------------------
# Standalone per-shape characterization figure.
# ---------------------------------------------------------------------------


def render_shape(shape, outdir, n, num_bins, seed):
    bins, counts, num_bins = char_input(shape, n, num_bins, seed)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    # Wrap the blurb so a long description does not run past the figure edge.
    blurb = "\n".join(
        textwrap.wrap(
            f"InputShape: {shape}   —   {SHAPE_BLURB.get(shape, '')}", width=150
        )
    )
    fig.suptitle(
        f"{blurb}\n(N={fmt_int(n)} samples, {fmt_bins(num_bins)} bins, seed={seed})",
        fontsize=12,
    )
    draw_distribution(axes[0], counts, num_bins)
    draw_rankfreq(axes[1], counts)
    draw_sequence(axes[2], bins, num_bins, shape=shape)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = os.path.join(outdir, f"{shape.replace(':', '_')}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out, num_bins, int(np.count_nonzero(counts)), int(counts.max())


def render_catalog(shapes, out, n, num_bins, seed, columns=3):
    """Render every shape into one high-resolution, zoomable catalog image."""
    rows = math.ceil(len(shapes) / columns)
    fig = plt.figure(figsize=(18 * columns, 5.2 * rows), constrained_layout=True)
    cards = fig.subfigures(rows, columns, squeeze=False)
    fig.suptitle(
        f"CUB histogram input characterizations (N={fmt_int(n)} samples, seed={seed})",
        fontsize=18,
    )

    for index, card in enumerate(cards.flat):
        if index >= len(shapes):
            card.set_visible(False)
            continue

        shape = shapes[index]
        bins, counts, shape_bins = char_input(shape, n, num_bins, seed)
        axes = card.subplots(1, 3)
        blurb = "\n".join(
            textwrap.wrap(
                f"InputShape: {shape} — {SHAPE_BLURB.get(shape, '')}", width=105
            )
        )
        card.suptitle(f"{blurb}\n({fmt_bins(shape_bins)} bins)", fontsize=10)
        draw_distribution(axes[0], counts, shape_bins)
        draw_rankfreq(axes[1], counts)
        draw_sequence(axes[2], bins, shape_bins, shape=shape)
        axes[0].set_title("value distribution (log y)", fontsize=9)
        axes[1].set_title(
            f"rank-frequency — {int(np.count_nonzero(counts))} occupied bins",
            fontsize=9,
        )
        axes[2].set_title("position in input sequence", fontsize=9)

    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--outdir",
        default="histogram_input_figs",
        help="output directory for the per-shape PNGs",
    )
    ap.add_argument(
        "--elements", type=int, default=CHAR_N, help="number of samples to characterize"
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=None,
        help="override the per-shape bin count (default: each shape's natural scale)",
    )
    ap.add_argument("--seed", type=int, default=CHAR_SEED, help="generator seed")
    ap.add_argument(
        "--shapes", nargs="*", default=SHAPES, help="subset of InputShapes to render"
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="also render all selected shapes into one catalog PNG",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(
        f"Characterizing {len(args.shapes)} shapes (N={fmt_int(args.elements)}, seed={args.seed})"
    )
    for shape in args.shapes:
        out, nb, occupied, hottest = render_shape(
            shape, args.outdir, args.elements, args.bins, args.seed
        )
        print(
            f"  {shape:<20} bins={fmt_bins(nb):<5} occupied={occupied:<6} hottest={fmt_int(hottest):<12} -> {out}"
        )
    if args.combined:
        combined = render_catalog(
            args.shapes,
            os.path.join(args.outdir, "all_input_characterizations.png"),
            args.elements,
            args.bins,
            args.seed,
        )
        print(f"Combined catalog -> {combined}")
    print(f"Done. {len(args.shapes)} figures under {args.outdir}")


if __name__ == "__main__":
    main()
