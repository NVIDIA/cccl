//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Apply partitioning operations on shapes to manipulate data subsets
 */

#include <cuda/experimental/__stf/places/tiled_partition.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

// Compute which part ID has the item at position "index"
__host__ __device__ size_t ref_tiling(size_t index, size_t tile_size, size_t nparts)
{
  // in which tile is this ?
  size_t tile_id = index / tile_size;

  // part which owns this tile
  return (tile_id % nparts);
}

int main()
{
  stream_ctx ctx;

  const size_t nparts    = 4;
  const size_t tile_size = 8;

  // Be sure to pick a number that is not a divisor to stress the tiling operator
  const size_t N = nparts * 3 * tile_size + 7;

  double Y[N];

  auto ly = ctx.logical_data(Y);

  // Init Y
  ctx.parallel_for(ly.shape(), ly.write())->*[=] _CCCL_DEVICE(size_t pos, auto sy) {
    sy(pos) = -1.0;
  };

  /*
   * We apply a tiling operator on the shape of ly, to work on subsets of ly.
   * For each subset, we put the id of the subset in the corresponding
   * entries of Y
   *
   * Note that these tasks are serialized as they perform a rw() on the
   * logical data as a whole.
   */
  for (size_t part_id = 0; part_id < nparts; part_id++)
  {
    ctx.parallel_for(tiled<tile_size>(ly.shape(), part_id, nparts), ly.rw())->*[=] _CCCL_DEVICE(size_t pos, auto sy) {
      sy(pos) = (double) part_id;
    };
  }

  bool checked   = false;
  bool* pchecked = &checked;

  /* Check the result on the host */
  ctx.parallel_for(exec_place::host, ly.shape(), ly.read())->*[=](size_t pos, slice<double> sy) {
    int expected = static_cast<int>(ref_tiling(pos, tile_size, nparts));
    int value    = (int) sy(pos);
    if (expected != value)
    {
      printf("POS %zu -> %d (expected %d)\n", pos, value, expected);
    }
    assert(expected == value);
    *pchecked = true;
  };

  ctx.finalize();

  // Ensure verification code did occur
  assert(checked);
}
