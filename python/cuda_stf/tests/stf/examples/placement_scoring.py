# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Placement scoring — choose a partition BEFORE committing memory.

Demonstrates:
  - **placement_evaluate** to score candidate mappings without allocating
  - why some natural-looking partitions are dramatically bad under
    page-granular placement, while others are perfect *in spite of* paging
  - **tensor-of-tiles** as the layout that makes multi-dimensional
    distributions page-exact

Physical placement is page-granular (typically 2 MiB): each page lands
entirely on ONE place, with the page's majority owner. Whether a partition
survives that rounding is decided by the length of its *ownership runs* —
the stretches of contiguous bytes with a single owner:

  runs >> page  ->  pages land where they belong: accuracy ~ 1.0
  runs << page  ->  every page mixes owners: up to half the bytes end up
                    remote to the place that computes on them, and every
                    access to them crosses the interconnect, every iteration

``accuracy`` (fraction of bytes local to their owner — byte-exact for the
structured partitions used here) is the score, and ``nallocs`` (physical
mapping runs) the fragmentation cost, both obtained without allocating a
byte. The four candidates below score a 256 MiB matrix:

  1. columns-blocked  1-D : ownership runs of 16 KiB    -> accuracy 0.5  BAD
  2. rows-blocked     1-D : two 128 MiB runs            -> accuracy 1.0  GOOD
  3. 2-D blocked, flat layout : runs of 16 KiB again    -> accuracy 0.5  BAD
  4. 2-D blocked, tensor-of-tiles (2 MiB tiles = pages) -> accuracy 1.0  GOOD

Candidates 1 and 3 look reasonable on paper — they are how one would shard
the matrix mathematically — but the run length, not the math, decides the
physical outcome. Candidate 4 shows the general escape: reorganize storage
into tiles sized to the page, and ANY distribution of whole tiles becomes
page-exact.

Scoring needs no VMM support and a single GPU: nothing is allocated.
"""

import cuda.stf._experimental as stf

MiB = 1024 * 1024

ROWS, COLS = 8192, 8192  # float32 -> 256 MiB
ELEMSIZE = 4
PAGE = 2 * MiB
ROW_BYTES = COLS * ELEMSIZE  # 32 KiB: one page holds 64 complete rows


def score(grid, partition):
    return stf.placement_evaluate(grid, partition, None, elemsize=ELEMSIZE, block_size=PAGE)


def report(name, s):
    print(
        f"  {name:34s} accuracy={s.accuracy:5.3f}  nallocs={s.nallocs:5d}  "
        f"bytes/place={[b // MiB for b in s.bytes_per_grid_index]} MiB"
    )
    return s


def main():
    stf.machine_init()

    # ---- 1-D candidates on a 2-place grid --------------------------------
    grid2 = stf.exec_place_grid.from_devices([0, 0])

    # BAD: block the contiguous (inner) axis. Each row is split between the
    # two owners, so ownership runs are COLS/2 * 4 B = 16 KiB — 128x smaller
    # than the 2 MiB page. Every page holds a 50/50 mix of both owners: no
    # placement can do better than getting half the bytes wrong.
    cols_blocked = stf.cute_partition.from_spec((ROWS, COLS), (None, ("blocked", 0)), (2,))
    s1 = report("1-D columns-blocked (inner axis)", score(grid2, cols_blocked))

    # GOOD: block the outer axis. Two contiguous 128 MiB runs, page-aligned:
    # every page belongs entirely to its owner.
    rows_blocked = stf.cute_partition.from_spec((ROWS, COLS), (("blocked", 0), None), (2,))
    s2 = report("1-D rows-blocked (outer axis)", score(grid2, rows_blocked))

    # ---- 2-D candidates on a 2x2 grid ------------------------------------
    grid4 = stf.exec_place_grid.create([stf.exec_place.device(0)] * 4, grid_dims=(2, 2))

    # BAD: the natural 2-D block distribution of the FLAT matrix. The row
    # halves alternate owners along every row, so the runs are 16 KiB again:
    # adding grid dimensions does not fix a layout problem.
    naive_2d = stf.cute_partition.from_spec(
        (ROWS, COLS), (("blocked", 0), ("blocked", 1)), (2, 2)
    )
    s3 = report("2-D blocked, flat layout", score(grid4, naive_2d))

    # GOOD in spite of paging: tensor-of-tiles. Storage is reorganized as
    # (tiles_y, tiles_x, tile_y, tile_x) with a 512x1024 float32 payload =
    # exactly one 2 MiB page per tile. Ownership runs now span whole tiles,
    # so the SAME 2-D blocked distribution becomes page-exact.
    tiles = (ROWS // 512, COLS // 1024)  # (16, 8) tiles
    tile_payload = (512, 1024)
    tensor_of_tiles = stf.cute_partition.from_spec(
        tiles + tile_payload,
        (("blocked", 0), ("blocked", 1), None, None),
        (2, 2),
    )
    s4 = report("2-D blocked, tensor-of-tiles", score(grid4, tensor_of_tiles))

    # The scores are byte-exact (structured partitions use the analytic /
    # census tiers), so the poster-child contrast can be asserted precisely.
    assert abs(s1.accuracy - 0.5) < 1e-12, "mixed pages: half the bytes are misplaced"
    assert s2.accuracy == 1.0 and s2.nallocs == 2, "two page-aligned runs"
    assert abs(s3.accuracy - 0.5) < 1e-12, "2-D on a flat layout is still mixed pages"
    assert s4.accuracy == 1.0, "whole-tile runs are page-exact under any tile distribution"
    assert s1.total_bytes == s2.total_bytes == s3.total_bytes == s4.total_bytes == ROWS * COLS * ELEMSIZE

    print(
        "\n  Same matrix, same page size: the run length decides. Score with\n"
        "  placement_evaluate() before allocating; misplaced bytes become\n"
        "  interconnect traffic on every access for the lifetime of the data."
    )


if __name__ == "__main__":
    main()
