#!/usr/bin/env python3
# Render the SMEM-capacity sweeps in the SAME format as the main-sweep
# algo_perf_figs: a folder per (transform, channels, sample type), one image per
# InputShape, a characterization top row, then a grid of per-element-count panels
# with GiB/s on Y and #bins on X. The only difference from the main charts: each
# line is a CAPACITY SETTING (cache slot count for Lever A; privatized SMEM cap for
# Lever B) instead of an algorithm, with the STATUS QUO drawn as a thick reference
# line. So bins, element count, and input shape are all visible -- nothing is
# aggregated away.
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the shared characterization helpers (co-located sibling module).
sys.path.insert(0, HERE)
import histogram_input_characterization as C  # noqa: E402

# Read the capacity-sweep JSONs from / write figs to $HIST_SWEEP_OUTDIR (default:
# cwd) -- this tracked script must not read or write inside the source tree.
DATADIR = os.environ.get("HIST_SWEEP_OUTDIR", ".")
FIGS = os.path.join(DATADIR, "capacity_figs")
SM_BYTES = 233472  # B200 SMEM/SM
OPTIN = 232448     # B200 opt-in SMEM/block

# label -> (transform, channels)
BINARY_META = {"even": ("even", "single"), "range": ("range", "single"),
               "multi_even": ("even", "multi"), "multi_range": ("range", "multi")}


def blk_per_sm(smem_bytes):
    return max(1, SM_BYTES // smem_bytes) if smem_bytes else 0


def fmt_elements(e):
    e = int(e)
    if e % (1 << 30) == 0:
        return f"{e >> 30}G"
    if e % (1 << 20) == 0:
        return f"{e >> 20}M"
    if e >= 1 << 30:
        return f"{e / (1 << 30):.2f}G"
    return f"{e / (1 << 20):.1f}M"


def _grid(series, sample, shape, elements_list, outpath, title, legend_title):
    """series: list of (label, color, lw, zorder, cells-dict, is_status_quo).
    cells-dict keyed 'sample|elements|bins|shape' -> gibs. One panel per element."""
    bins_c, counts_c, char_bins = C.char_input(shape)
    ncols = 3
    nperf = len(elements_list)
    nrows = 1 + (nperf + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.6 * ncols, 4.1 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    head = "\n".join(textwrap.wrap(title, width=125))
    fig.suptitle(head + f"\ntop: input characterization (N={C.fmt_int(C.CHAR_N)}, bins={C.fmt_bins(char_bins)})"
                 f"   below: GiB/s vs #bins per input size — one line per {legend_title}", fontsize=12)
    C.draw_distribution(fig.add_subplot(gs[0, 0]), counts_c, char_bins)
    C.draw_sequence(fig.add_subplot(gs[0, 1]), bins_c, char_bins, shape=shape)
    # Series tuples are (lbl, color, lw, zorder, cells, sq[, marker]); reference
    # algorithm lines pass a 7th element (their algo marker) so they read like the
    # main sweep, while the swept cache-size lines keep the default "o".
    def _unpack(s):
        lbl, c, lw, z, cells, sq = s[:6]
        marker = s[6] if len(s) > 6 else "o"
        return lbl, c, lw, z, cells, sq, marker

    lax = fig.add_subplot(gs[0, 2]); lax.axis("off")
    handles = [plt.Line2D([0], [0], color=c, lw=lw, ls=("--" if sq else "-"),
                          marker=mk, ms=(11 if mk == "*" else 6), label=lbl)
               for (lbl, c, lw, z, cells, sq, mk) in (_unpack(s) for s in series)]
    lax.legend(handles=handles, loc="center", fontsize=8, title=legend_title, frameon=True)

    for i, elems in enumerate(elements_list):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        any_pts = False
        for (lbl, color, lw, z, cells, sq, marker) in (_unpack(s) for s in series):
            pts = []
            for k, g in cells.items():
                s, e, b, sh = k.split("|")
                if s == sample and int(e) == elems and sh == shape:
                    pts.append((int(b), g))
            pts.sort()
            if pts:
                any_pts = True
                xb = [p[0] for p in pts]; yv = [p[1] for p in pts]
                ax.plot(xb, yv, color=color, lw=lw, marker=marker, ms=(11 if marker == "*" else 5),
                        zorder=z, ls=("--" if sq else "-"), alpha=0.9)
        ax.set_title(f"N = {fmt_elements(elems)} elements", fontsize=9)
        ax.set_xlabel("# bins"); ax.set_ylabel("GiB/s")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        allb = sorted({int(k.split('|')[2]) for s in series for k in s[4]
                       if k.split('|')[0] == sample and int(k.split('|')[1]) == elems and k.split('|')[3] == shape})
        if allb:
            ax.set_xticks(allb)
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: C.fmt_bins(int(round(v)))))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        if any_pts:
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "no data\n(cells skipped:\noverflow / can't launch)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


def cap_color(idx, n):
    return plt.cm.viridis(0.12 + 0.76 * idx / max(1, n - 1))


# Lever-A two-family styling so the swept cache and the reference algorithms are
# visually separable at a glance.
#   * CACHED (the cache-size series of the algo being swept): one COOL family --
#     a Blues ramp, light = small cache, dark = large; "auto" is a blue STAR.
#   * REF (the other algorithms, cache-size-irrelevant): one WARM family --
#     distinct orange/red/brown/magenta hues, each with its own marker.
def blue_for(ls, sizes):
    if len(sizes) <= 1:
        return plt.cm.Blues(0.78)
    j = sizes.index(ls)
    return plt.cm.Blues(0.42 + 0.5 * j / (len(sizes) - 1))


REF_WARM = {  # warm family for reference algorithms (color, marker, label)
    "direct_atomic_no_cache": ("#e6550d", "v", "[ref] no cache (direct atomics)"),
    "gmem_priv_gather": ("#d62728", "^", "[ref] gmem gather-merge"),
    "hybrid_single_pass": ("#8c2d04", "D", "[ref] hybrid SMEM+GMEM"),
    "direct_atomic_single_probe": ("#e377c2", "s", "[ref] single-probe cache"),
    "direct_atomic_cuckoo": ("#fdae6b", "P", "[ref] cuckoo cache"),
}


# ----------------------------- Lever A: cache slots -----------------------------
def plot_lever_a():
    data = json.load(open(os.path.join(DATADIR, "cache_slots_results.json")))
    for algo in ("direct_atomic_cuckoo", "direct_atomic_single_probe"):
        for label, av in data.items():
            transform, channels = BINARY_META[label]
            # gather this algo's slot settings that actually ran the forced algo
            runs = {}  # launched_slots -> rec   (dedupe; prefer request==launched)
            auto = None
            for tag, rec in av.items():
                a, s = tag.split("@")
                if a != algo or not rec["ran_forced"] or not rec["cells"]:
                    continue
                if s == "auto":
                    auto = rec
                else:
                    ls = int(rec["launched_slots"]); req = int(s)
                    if ls not in runs or (req == ls):
                        runs[ls] = rec
            if auto is None:
                continue
            # samples + elements present
            samples = sorted({k.split("|")[0] for k in auto["cells"]})
            elements = sorted({int(k.split("|")[1]) for k in auto["cells"]})
            shapes = sorted({k.split("|")[3] for k in auto["cells"]})
            distinct = sorted(runs)
            # Reference algorithms (no_cache / gather / hybrid), measured once each
            # via the "<algo>@ref" tags -- shown on every figure as flat comparison
            # lines so each cache-size curve is judged against the alternatives.
            ref = {}
            for tag, rec in av.items():
                a, s = tag.split("@")
                if s == "ref" and rec["cells"]:
                    ref[a] = rec["cells"]
            for sample in samples:
                folder = os.path.join(FIGS, f"lever_a_cache_{algo.replace('direct_atomic_', '')}",
                                      f"{transform}_{channels}_{sample}")
                for shape in shapes:
                    series = []
                    asm = auto["launched_smem_bytes"]
                    cname = algo.replace("direct_atomic_", "")
                    # 1) CACHED family (cool / Blues): one line per cache size, light
                    #    = small cache -> dark = large. "auto" (the status quo) is a
                    #    blue STAR so it stands out WITHIN the family (no black line).
                    series.append((f"{cname} cache: auto = {auto['launched_slots']} slots · "
                                   f"{asm//1024}KB · {blk_per_sm(asm)}blk/SM (status quo)",
                                   blue_for(auto["launched_slots"], distinct), 3.0, 10, auto["cells"], False, "*"))
                    for ls in distinct:
                        if ls == auto["launched_slots"]:
                            continue  # the auto star already covers this size
                        sm = runs[ls]["launched_smem_bytes"]
                        series.append((f"{cname} cache: {ls} slots · {sm//1024}KB · {blk_per_sm(sm)}blk/SM",
                                       blue_for(ls, distinct), 1.8, 4, runs[ls]["cells"], False, "o"))
                    # 2) REF family (warm): the OTHER algorithms (cache-size
                    #    irrelevant), distinct warm hues + markers.
                    for a in ALGO_ORDER:
                        if a == algo or a not in ref:
                            continue
                        color, marker, lbl = REF_WARM[a]
                        series.append((lbl, color, 1.7, 3, ref[a], False, marker))
                    title = (f"LEVER A (cache size) — {transform.upper()} · {channels}-channel · {sample} · "
                             f"sweeping {algo.replace('direct_atomic_', '')} cache — InputShape: {shape} "
                             f"({C.SHAPE_BLURB.get(shape, '')})")
                    out = os.path.join(folder, f"{shape.replace(':', '_')}.png")
                    _grid(series, sample, shape, elements, out, title,
                          "cache size (this algo) + [ref] other algorithms")
    print(f"[A] lever_a_cache_* folders written under {FIGS}")


# Algorithm styling (mirrors the main sweep's algo_perf chart).
ALGO_STYLE = {
    "direct_atomic_cuckoo": ("#1f77b4", "o", "cuckoo cache"),
    "direct_atomic_single_probe": ("#2ca02c", "s", "single-probe cache"),
    "direct_atomic_no_cache": ("#ff7f0e", "v", "no cache (direct atomics)"),
    "gmem_priv_gather": ("#9467bd", "^", "gmem gather-merge"),
    "hybrid_single_pass": ("#d62728", "D", "hybrid SMEM+GMEM"),
}
ALGO_ORDER = list(ALGO_STYLE)
PRIV_COLOR = "#000000"
CAPS = [16384, 24576, 32768, 49152]


# ----------------------------- Lever B: privatized cap --------------------------
def plot_lever_b():
    """Redesigned: priv_dynamic's throughput at a bin count is INDEPENDENT of which
    cap-variant ran it (verified) -- a larger cap only extends HOW FAR the on-chip
    kernel reaches, not its speed. So show ONE smem_priv_dynamic line, plotted only
    at bins it can actually serve on-chip (bins <= the largest cap, 49152), and
    overlay ALL the high-bin algorithms across ALL bin counts for comparison. The
    cap boundaries are vertical markers showing how far right each cap would let
    priv_dynamic run."""
    data = json.load(open(os.path.join(DATADIR, "priv_cap_results.json")))
    by_label = {}  # label -> {"priv": {cell:gibs}, algo: {cell:gibs}, ...}
    for tag, rec in data.items():
        label, algo = rec["label"], rec["algo"]
        d = by_label.setdefault(label, {})
        if algo == "natural":
            # priv_dynamic perf is cap-independent, so keep the best value across cap
            # variants for each cell. CRITICAL: a cell counts as priv_dynamic only if
            # its bin is <= THIS binary's cap (rec["cap"]). The launch-set is
            # per-invocation, so the stock cap16384 binary lists 'smem_priv_dynamic'
            # (from its <=16384 cells) while its 24576+ cells actually ran the high-bin
            # path -- gating on rec["cap"] (not max cap) excludes those.
            priv = d.setdefault("priv", {})
            for cell, g in rec["cells"].items():
                bins = int(cell.split("|")[2])
                if bins <= rec["cap"] and ("smem_priv_dynamic" in rec["launched"]):
                    if g > priv.get(cell, 0):
                        priv[cell] = g
        else:
            d.setdefault(algo, {}).update(rec["cells"])

    for label, d in by_label.items():
        transform, channels = BINARY_META[label]
        priv = d.get("priv", {})
        algos_present = [a for a in ALGO_ORDER if d.get(a)]
        # discover axes from whatever data exists
        allcells = list(priv) + [c for a in algos_present for c in d[a]]
        if not allcells:
            continue
        samples = sorted({c.split("|")[0] for c in allcells})
        elements = sorted({int(c.split("|")[1]) for c in allcells})
        shapes = sorted({c.split("|")[3] for c in allcells},
                        key=lambda s: (C.SHAPES.index(s) if s in C.SHAPES else 99, s))
        for sample in samples:
            folder = os.path.join(FIGS, "lever_b_cap", f"{transform}_{channels}_{sample}")
            for shape in shapes:
                title = (f"LEVER B (privatized SMEM cap) — {transform.upper()} · {channels}-channel · {sample} — "
                         f"InputShape: {shape} ({C.SHAPE_BLURB.get(shape, '')})")
                out = os.path.join(folder, f"{shape.replace(':', '_')}.png")
                _render_lever_b(priv, d, algos_present, sample, shape, elements, out, title, channels)
    print(f"[B] lever_b_cap folders written under {FIGS}")


def _series_for(cells, sample, elems, shape):
    pts = sorted((int(k.split("|")[2]), g) for k, g in cells.items()
                 if k.split("|")[0] == sample and int(k.split("|")[1]) == elems and k.split("|")[3] == shape)
    return [p[0] for p in pts], [p[1] for p in pts]


def _render_lever_b(priv, algo_data, algos_present, sample, shape, elements_list, outpath, title, channels):
    bins_c, counts_c, char_bins = C.char_input(shape)
    ncols = 3
    nperf = len(elements_list)
    nrows = 1 + (nperf + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.8 * ncols, 4.2 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    head = "\n".join(textwrap.wrap(title, width=125))
    fig.suptitle(head + f"\ntop: input characterization (N={C.fmt_int(C.CHAR_N)}, bins={C.fmt_bins(char_bins)})"
                 "   below: GiB/s vs #bins — smem_priv_dynamic (on-chip, bins<=cap) vs every high-bin algorithm",
                 fontsize=12)
    C.draw_distribution(fig.add_subplot(gs[0, 0]), counts_c, char_bins)
    C.draw_sequence(fig.add_subplot(gs[0, 1]), bins_c, char_bins, shape=shape)
    lax = fig.add_subplot(gs[0, 2]); lax.axis("off")
    handles = [plt.Line2D([0], [0], color=PRIV_COLOR, lw=3.0, marker="*", ms=10,
                          label="smem_priv_dynamic (on-chip)")]
    handles += [plt.Line2D([0], [0], color=ALGO_STYLE[a][0], marker=ALGO_STYLE[a][1], lw=1.8,
                           label=ALGO_STYLE[a][2]) for a in algos_present]
    handles.append(plt.Line2D([0], [0], color="gray", lw=1, ls=":", label="cap boundary (priv reaches ≤ here)"))
    lax.legend(handles=handles, loc="center", fontsize=9, title="series", frameon=True)

    for i, elems in enumerate(elements_list):
        ax = fig.add_subplot(gs[1 + i // ncols, i % ncols])
        any_pts = False
        # cap-boundary vlines (only those <= max bin shown)
        for c in CAPS:
            ax.axvline(c, color="gray", ls=":", lw=0.9, alpha=0.6, zorder=1)
            ax.annotate(f"cap {c//1024}K\n{blk_per_sm(c*4)}blk/SM", (c, 0.99), xycoords=("data", "axes fraction"),
                        fontsize=6, color="gray", ha="center", va="top", rotation=90)
        # high-bin algorithms across ALL bins
        for a in algos_present:
            xb, yv = _series_for(algo_data[a], sample, elems, shape)
            if xb:
                any_pts = True
                color, marker, _ = ALGO_STYLE[a]
                ax.plot(xb, yv, color=color, marker=marker, ms=5, lw=1.7, alpha=0.85, zorder=3)
        # priv_dynamic — ONLY at bins it serves on-chip (bins <= max cap)
        xb, yv = _series_for(priv, sample, elems, shape)
        if xb:
            any_pts = True
            ax.plot(xb, yv, color=PRIV_COLOR, marker="*", ms=11, lw=3.0, alpha=0.95, zorder=6)
        ax.set_title(f"N = {fmt_elements(elems)} elements", fontsize=9)
        ax.set_xlabel("# bins"); ax.set_ylabel("GiB/s")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        allb = sorted({int(k.split("|")[2]) for src in [priv] + [algo_data[a] for a in algos_present]
                       for k in src if k.split("|")[0] == sample and int(k.split("|")[1]) == elems
                       and k.split("|")[3] == shape})
        if allb:
            ax.set_xticks(allb)
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: C.fmt_bins(int(round(v)))))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
        if any_pts:
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "no data\n(cells skipped:\noverflow / can't launch)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color="gray")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_lever_a()
    plot_lever_b()
    print("done")
