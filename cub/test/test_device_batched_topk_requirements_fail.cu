// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ERR err 0:1:2:3:4:5

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

// Verifies that cub::DeviceBatchedTopK rejects, at compile time, the determinism and tie_break requirement
// combinations that the public contract marks as ill-formed (see docs/cub/device_topk_requirements.rst):
//   * determinism and tie_break must be acknowledged together, both specified or both omitted to take the default
//   * an explicit tie_break of prefer_smaller_index or prefer_larger_index pins the result set across GPUs and so
//     requires determinism::gpu_to_gpu (it cannot be paired with not_guaranteed or run_to_run)
// Each variant exercises one rejected cell of that contract table.

int main()
{
  namespace ex = cuda::execution;

  // Per-segment key iterators (iterator-of-iterators). The keys-only entry point ignores value iterators.
  int** d_keys_in    = nullptr;
  int** d_keys_out   = nullptr;
  auto segment_sizes = cuda::args::constant<8>{};
  auto k_arg         = cuda::args::constant<3>{};
  auto num_segments  = cuda::args::immediate{cuda::std::int64_t{2}};

#if TEST_ERR == 0 // determinism specified without a paired tie_break
  auto requirements = ex::require(ex::determinism::not_guaranteed, ex::output_ordering::unsorted);
  // expected-error-0 {{"must be acknowledged together"}}
#elif TEST_ERR == 1 // tie_break specified without a paired determinism
  auto requirements = ex::require(ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-1 {{"must be acknowledged together"}}
#elif TEST_ERR == 2 // explicit tie_break with not_guaranteed
  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-2 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 3 // explicit tie_break with not_guaranteed
  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_larger_index, ex::output_ordering::unsorted);
  // expected-error-3 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 4 // explicit tie_break with run_to_run
  auto requirements =
    ex::require(ex::determinism::run_to_run, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted);
  // expected-error-4 {{"pins the result set across GPUs and therefore requires"}}
#elif TEST_ERR == 5 // explicit tie_break with run_to_run
  auto requirements =
    ex::require(ex::determinism::run_to_run, ex::tie_break::prefer_larger_index, ex::output_ordering::unsorted);
  // expected-error-5 {{"pins the result set across GPUs and therefore requires"}}
#endif

  auto env                  = cuda::std::execution::env{requirements};
  size_t temp_storage_bytes = 0;
  auto error                = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
