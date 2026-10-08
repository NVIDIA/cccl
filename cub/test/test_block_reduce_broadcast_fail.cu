// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce.cuh>

#include <cuda/std/functional>

// %PARAM% TEST_ALGORITHM algo 0:1:2

__global__ void unsupported_broadcast(int* output)
{
  constexpr cub::BlockReduceAlgorithm algorithm =
    TEST_ALGORITHM == 0 ? cub::BLOCK_REDUCE_RAKING
    : TEST_ALGORITHM == 1
      ? cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
      : cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
  using block_reduce_t = cub::BlockReduce<int, 128, algorithm>;
  __shared__ typename block_reduce_t::TempStorage storage;
  // expected-error {{"ReduceBroadcast requires BLOCK_REDUCE_WARP_REDUCTIONS"}}
  output[threadIdx.x] = block_reduce_t(storage).ReduceBroadcast(1, cuda::std::plus<>{});
}

int main() {}
