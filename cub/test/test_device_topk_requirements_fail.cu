// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% TEST_ERR err 0:1:2

#include <cub/device/device_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/std/execution>

#include <iostream>

// Verifies that cub::DeviceTopK rejects, at compile time, tie-break requests it cannot honor. DeviceTopK implements
// only the fully opted-out configuration (determinism::not_guaranteed + output_ordering::unsorted) and no tie-breaking
// yet, so a prefer_smaller_index / prefer_larger_index tie-break would otherwise be silently ignored. Each variant
// checks a single diagnostic:
//
//   * variant 0: tie_break placed in the environment outside require(...), where it has no effect
//   * variants 1-2: an explicit prefer_smaller_index / prefer_larger_index together with not_guaranteed
//
// Omitting tie_break (the pre-3.5 `require(not_guaranteed, unsorted)` form) and tie_break::unspecified stay accepted.
// The determinism/tie_break pairing rule enforced by cub::DeviceBatchedTopK is deferred to CCCL 4.0 for DeviceTopK.

int main()
{
  namespace ex = cuda::execution;

  int* d_keys_in  = nullptr;
  int* d_keys_out = nullptr;

#if TEST_ERR == 0 // tie_break outside require(...)
  auto env = cuda::std::execution::env{
    ex::require(ex::determinism::not_guaranteed, ex::output_ordering::unsorted), ex::tie_break::prefer_smaller_index};
  // expected-error-0 {{"Tie-break should be used inside cuda::execution::require"}}
#elif TEST_ERR == 1 // explicit tie_break with not_guaranteed
  auto env = cuda::std::execution::env{
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_smaller_index, ex::output_ordering::unsorted)};
  // expected-error-1 {{"only supports cuda::execution::tie_break::unspecified"}}
#elif TEST_ERR == 2 // explicit tie_break with not_guaranteed
  auto env = cuda::std::execution::env{
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::prefer_larger_index, ex::output_ordering::unsorted)};
  // expected-error-2 {{"only supports cuda::execution::tie_break::unspecified"}}
#endif

  const int num_items       = 0;
  const int k               = 0;
  size_t temp_storage_bytes = 0;
  auto error = cub::DeviceTopK::MaxKeys(nullptr, temp_storage_bytes, d_keys_in, d_keys_out, num_items, k, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTopK::MaxKeys failed with status: " << error << '\n';
  }
}
