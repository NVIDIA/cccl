//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Placement scoring: choose a partition BEFORE committing memory
 *
 * Physical placement is page-granular (typically 2 MiB): each page lands
 * entirely on ONE place, with the page's majority owner. Whether a partition
 * survives that rounding is decided by the length of its ownership runs (the
 * stretches of contiguous bytes with a single owner):
 *
 *   runs >> page  ->  pages land where they belong: accuracy ~ 1.0
 *   runs << page  ->  every page mixes owners: up to half the bytes end up
 *                     remote to the place computing on them, and every access
 *                     to them crosses the interconnect, every iteration
 *
 * evaluate_localized_placement() scores a candidate mapping without
 * allocating a byte: `accuracy()` is the fraction of bytes local to their
 * owner (byte-exact for the structured partitions used here), and `nallocs`
 * the physical mapping fragmentation. Four candidates score one 256 MiB
 * row-major matrix (dim4 dimension 0 is the contiguous axis):
 *
 *   1. columns-blocked 1-D (splits the contiguous axis): 16 KiB runs -> 0.5
 *   2. rows-blocked    1-D (splits the outer axis): 128 MiB runs     -> 1.0
 *   3. 2-D blocked over a 2x2 grid, flat layout: 16 KiB runs again   -> 0.5
 *   4. 2-D blocked, tensor-of-tiles (2 MiB tiles = pages)            -> 1.0
 *
 * Candidates 1 and 3 are how one would shard the matrix mathematically, but
 * the run length, not the math, decides the physical outcome. Candidate 4 is
 * the general escape: reorganize storage into page-sized tiles and ANY
 * distribution of whole tiles becomes page-exact.
 *
 * Scoring allocates nothing, so this runs on a single GPU (the grids below
 * repeat device 0) and needs no VMM support.
 */

#include <cuda/experimental/stf.cuh>

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace cuda::experimental::stf;

namespace
{

constexpr size_t MiB  = 1024 * 1024;
constexpr size_t ROWS = 8192;
constexpr size_t COLS = 8192; // float payload: 256 MiB, 32 KiB rows
constexpr size_t PAGE = 2 * MiB;

localized_stats report(const char* name, const exec_place& grid, const localized_stats& s)
{
  printf("  %-36s accuracy=%5.3f  nallocs=%5zu  MiB/position=[", name, s.accuracy(), s.nallocs);
  const size_t grid_size = grid.get_dims().size();
  for (size_t i = 0; i < grid_size; i++)
  {
    const auto it   = s.bytes_per_grid_index.find(i);
    const size_t mb = (it == s.bytes_per_grid_index.end()) ? 0 : it->second / MiB;
    printf("%s%zu", i ? ", " : "", mb);
  }
  printf("]\n");
  return s;
}

exec_place make_dev0_grid(size_t nplaces, dim4 dims)
{
  ::std::vector<exec_place> places(nplaces, exec_place::device(0));
  return make_grid(::std::move(places), dims);
}

//! Score a candidate mapping at 2 MiB page granularity, allocating nothing.
template <typename Partition>
localized_stats score(const exec_place& grid, const Partition& part)
{
  return evaluate_localized_placement(
    grid, part, sizeof(float), ::cuda::experimental::places::localized_placement_default_probes, PAGE);
}

} // namespace

int main()
{
  // Scoring is placement-only: grids of repeated device-0 places make the
  // example single-GPU runnable; on a real machine, use all_devices().
  auto grid2 = make_dev0_grid(2, dim4(2));
  auto grid4 = make_dev0_grid(4, dim4(2, 2));

  // BAD: block the contiguous axis. Each row is split between the owners, so
  // ownership runs are COLS/2 floats = 16 KiB, 128x smaller than the page:
  // every page holds a 50/50 mix and half the bytes are misplaced whatever
  // the page's owner is.
  auto cols_blocked = make_partition(dim4(COLS, ROWS), partition_spec{blocked<0>, whole}, dim4(2));
  auto s1 = report("1-D columns-blocked (inner axis)", grid2, score(grid2, cols_blocked));

  // GOOD: block the outer axis. Two contiguous, page-aligned 128 MiB runs.
  auto rows_blocked = make_partition(dim4(COLS, ROWS), partition_spec{whole, blocked<0>}, dim4(2));
  auto s2 = report("1-D rows-blocked (outer axis)", grid2, score(grid2, rows_blocked));

  // BAD: the natural 2-D block distribution of the FLAT matrix. The row
  // halves alternate owners along every row: 16 KiB runs again. Adding grid
  // dimensions does not fix a layout problem.
  auto naive_2d = make_partition(dim4(COLS, ROWS), partition_spec{blocked<0>, blocked<1>}, dim4(2, 2));
  auto s3 = report("2-D blocked, flat layout", grid4, score(grid4, naive_2d));

  // GOOD in spite of paging: tensor-of-tiles. Storage reorganized as
  // (tile_x, tile_y, tiles_x, tiles_y) with a 1024x512 float payload =
  // exactly one 2 MiB page per tile: ownership runs span whole tiles, so the
  // SAME 2-D blocked distribution becomes page-exact.
  constexpr size_t TILE_X = 1024, TILE_Y = 512;
  auto tensor_of_tiles = make_partition(
    dim4(TILE_X, TILE_Y, COLS / TILE_X, ROWS / TILE_Y),
    partition_spec{whole, whole, blocked<0>, blocked<1>},
    dim4(2, 2));
  auto s4 = report("2-D blocked, tensor-of-tiles", grid4, score(grid4, tensor_of_tiles));

  // The structured tiers are byte-exact, so the contrast verifies precisely
  // (explicit checks, not asserts: they must hold in release builds too).
  const size_t total = ROWS * COLS * sizeof(float);
  auto expect        = [](bool ok, const char* what) {
    if (!ok)
    {
      fprintf(stderr, "Verification FAILED: %s\n", what);
      exit(1);
    }
  };
  expect(::std::fabs(s1.accuracy() - 0.5) < 1e-12, "mixed pages misplace half the bytes");
  expect(s2.accuracy() == 1.0 && s2.nallocs == 2, "two page-aligned runs");
  expect(::std::fabs(s3.accuracy() - 0.5) < 1e-12, "2-D on a flat layout is still mixed pages");
  expect(s4.accuracy() == 1.0, "whole-tile runs are page-exact under any tile distribution");
  expect(s1.total_bytes == total && s2.total_bytes == total && s3.total_bytes == total && s4.total_bytes == total,
         "every candidate scored the same matrix");

  printf("\n  Same matrix, same page size: the run length decides. Score with\n"
         "  evaluate_localized_placement() before allocating; misplaced bytes\n"
         "  become interconnect traffic on every access for the data's lifetime.\n");

  return 0;
}
