// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// [block-merge-example]
#include <cub/block/block_merge.cuh>

#include <cuda/std/functional>

// Each tile contains sixteen consecutive sorted runs of 32 keys.
_CCCL_KERNEL_ATTRIBUTES void merge_sorted_runs(const int* input, int* output)
{
  using Merge = cub::BlockMerge<int, 128, 4, 16>;
  __shared__ Merge::TempStorage storage;
  __shared__ int keys[128 * 4 + 1];
  __shared__ int offsets[17];

  const int tid  = static_cast<int>(threadIdx.x);
  const int base = static_cast<int>(blockIdx.x) * 512;
  for (int i = tid; i < 512; i += 128)
  {
    keys[i] = input[base + i];
  }
  if (tid <= 16)
  {
    offsets[tid] = tid * 32;
  }
  if (tid == 0)
  {
    keys[512] = 0; // Initialize the extra readable slot after the live keys.
  }
  __syncthreads();

  int items[4];
  Merge(storage).Merge(keys, offsets, 16, items, cuda::std::less<>{});
  __syncthreads(); // Before reusing shared storage.

  // Thread t owns merged ranks [t * 4, (t + 1) * 4).
  for (int i = 0; i < 4; ++i)
  {
    output[base + tid * 4 + i] = items[i];
  }
}
// [block-merge-example]

#include <thrust/detail/raw_pointer_cast.h>

#include <algorithm>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/generators.h>
#include <c2h/vector.h>

CUB_TEST("BlockMerge API example", "[block][merge][api]", CUB_SMALL)
{
  constexpr int tiles = 3;
  constexpr int count = 512;
  c2h::device_vector<int> input(tiles * count);
  c2h::gen(C2H_SEED(2), input, -31, 31);
  c2h::host_vector<int> sorted_runs = input;
  for (int run = 0; run < tiles * 16; ++run)
  {
    std::sort(sorted_runs.begin() + run * 32, sorted_runs.begin() + (run + 1) * 32);
  }
  input = sorted_runs;

  c2h::host_vector<int> expected = sorted_runs;
  for (int tile = 0; tile < tiles; ++tile)
  {
    std::sort(expected.begin() + tile * count, expected.begin() + (tile + 1) * count);
  }
  c2h::device_vector<int> output(input.size());
  merge_sorted_runs<<<tiles, 128>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(expected == output);
}
