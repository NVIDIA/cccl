#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: BSD-3
"""Re-derive the `default` (selector) column of a histogram sweep JSON under a NEW
single-channel selector, WITHOUT re-running any benchmarks.

This is valid only because the selector change (commit 0ec99ca6c6) touched
`select_algorithm` ONLY -- no kernel, agent, or launch code changed -- so every
algorithm's measured GiB/s is unchanged. The sweep already measured every forced
algorithm at every high-bin cell, and `default[cell]` was verified to equal
`forced[picked_algo][cell]` to within ~0.3% (median ratio 1.0000) in the source data.
So the new selector's `default` curve is exactly the forced column it would now pick.

We rewrite, per single-channel cell (binary `even` / `range`):
  - `default[cell]`   -> the forced column the NEW selector picks
  - `_selected[cell]` -> the launch-tag string the plotter maps to a 3-letter tag

Multi-channel binaries (`multi_even` / `multi_range`) are LEFT UNTOUCHED: the #44
change to the multi-channel arm was a pure constant rename (same value, same
conditions), so their selector picks are unchanged.

Cells at or below the on-chip cap (selector ran smem_privatized) are left untouched:
they have no forced columns and the new selector still runs smem_privatized there.

Usage:
  python retune_default_from_forced.py --in  <orig>/algo_sweep_full.json \
                                       --out <new>/algo_sweep_full.json
The output dir's figures are then regenerated with the existing histogram_algo_perf.py.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# --- NEW single-channel high-bin selector (mirrors select_algorithm in
# dispatch_histogram.cuh after commit 0ec99ca6c6). Pixel thresholds in pixels. ---
HYBRID_CAP_TIER_MAX_BINS = 65536
HYBRID_MID_TIER_MAX_BINS = 131072
CAP_TIER_AMORTIZE_PIXELS = 1 << 24  # 16M, both transforms
MID_TIER_AMORTIZE_PIXELS_EVEN = 1 << 24  # 16M
MID_TIER_AMORTIZE_PIXELS_RANGE = 1 << 26  # 64M
# Below this the selector keeps the histogram on chip (smem_privatized) and there is
# no forced column to substitute. Matches the run's recorded on-chip routing, which is
# the per-arch byte budget; for this B200 sweep single-channel that is 32768 on chip,
# 65536+ high-bin. We detect "high-bin" structurally: a cell is high-bin iff it has at
# least one forced column present (the sweep only emits forced columns above the cap).
SINGLE_CHANNEL_BINARIES = {"even": True, "range": False}  # name -> is_even


def new_single_channel_pick(is_even: bool, bins: int, pixels: int) -> str:
    """The algorithm enum name the NEW selector returns for a single-channel high-bin
    cell. Only called for cells the sweep treated as high-bin (forced columns exist)."""
    if bins <= HYBRID_CAP_TIER_MAX_BINS:
        return "gmem_privatized_nocache" if pixels >= CAP_TIER_AMORTIZE_PIXELS else "direct_single_probe"
    if bins <= HYBRID_MID_TIER_MAX_BINS:
        amortize = MID_TIER_AMORTIZE_PIXELS_EVEN if is_even else MID_TIER_AMORTIZE_PIXELS_RANGE
        return "gmem_privatized_nocache" if pixels >= amortize else "direct_single_probe"
    return "direct_single_probe"


def perf_column_for(pick: str) -> str:
    """Map a selector return value to the perf column that actually runs. For
    single-channel, dispatch runs the hybrid member of gmem_privatized_nocache (the
    smem_split>0 member), recorded under the `hybrid` column; the pure-gather member
    (`gmem_privatized_nocache` column) only runs when explicitly forced."""
    return "hybrid" if pick == "gmem_privatized_nocache" else pick


def launch_tag_for(pick: str) -> str:
    """The `_selected` launch-tag string the plotter expects (it maps this to a
    3-letter tag via ALGO_TAG). The hybrid member reports the `:hybrid` suffix."""
    return "gmem_privatized_nocache:hybrid" if pick == "gmem_privatized_nocache" else pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="source algo_sweep_full.json")
    ap.add_argument("--out", required=True, help="destination algo_sweep_full.json (default+_selected rewritten)")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    ratios_checked = []  # sanity: default vs forced agreement on UNCHANGED picks
    changed = 0
    total_high = 0

    for blabel, is_even in SINGLE_CHANNEL_BINARIES.items():
        if blabel not in data:
            continue
        cells = data[blabel]
        default = cells.get("default", {})
        selected = cells.setdefault("_selected", {})
        for key in list(default):
            sample, elements, bins, shape = key.split("|")
            bins_i = int(bins)
            pixels = int(elements)  # 1 channel: pixels == elements
            # A cell is in the high-bin region (where the selector chooses among
            # hybrid / direct-atomic) IFF its RECORDED `_selected` is not
            # smem_privatized. The on-chip tier (<= the per-arch byte-budget cap, which
            # on this B200 run puts 32768 on chip) runs smem_privatized and is unchanged
            # by the #44 high-bin retune -- skip it. Using the measured `_selected`
            # (ground truth from the actual binary) avoids re-deriving the on-chip cap.
            recorded = selected.get(key)
            if recorded is None or recorded == "smem_privatized":
                continue
            total_high += 1
            pick = new_single_channel_pick(is_even, bins_i, pixels)
            col = perf_column_for(pick)
            newval = cells.get(col, {}).get(key)
            if newval is None or newval <= 0:
                # The picked algorithm has no measurement here (shouldn't happen for a
                # legal pick). Leave the cell as-is and flag it.
                print(f"  !! {blabel} {key}: picked {pick} ({col}) has no value; left unchanged", flush=True)
                continue
            # Sanity: record how far the OLD default was from this NEW value (info only).
            oldval = default[key]
            if oldval > 0:
                ratios_checked.append(newval / oldval)
            default[key] = newval
            selected[key] = launch_tag_for(pick)
            changed += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(args.out, "w"), indent=1)
    print(f"rewrote default+_selected for {changed}/{total_high} single-channel high-bin cells "
          f"(even/range); multi_* untouched", flush=True)
    if ratios_checked:
        print(f"new/old default ratio over rewritten cells: median={statistics.median(ratios_checked):.3f} "
              f"min={min(ratios_checked):.3f} max={max(ratios_checked):.3f} "
              f"(geomean={math.exp(statistics.fmean(map(math.log, ratios_checked))):.3f})", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
