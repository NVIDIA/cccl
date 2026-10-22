// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

struct wide_value
{
  ::cuda::std::int64_t first;
  ::cuda::std::int64_t second;
  ::cuda::std::int64_t third;
};

int main()
{
  namespace ex = cuda::execution;

  int** d_keys_in           = nullptr;
  int** d_keys_out          = nullptr;
  wide_value** d_values_in  = nullptr;
  wide_value** d_values_out = nullptr;
  auto segment_sizes        = cuda::args::constant<2048>{};
  auto k_arg                = cuda::args::constant<2048>{};
  auto num_segments         = cuda::args::immediate{cuda::std::int64_t{2}};
  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::unspecified, ex::output_ordering::sorted);
  auto env = cuda::std::execution::env{requirements};

  // expected-error {{"sorted output cannot cover the statically-known maximum output size"}}
  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxPairs(
    nullptr,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k_arg,
    num_segments,
    env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxPairs failed with status: " << error << '\n';
  }
}
