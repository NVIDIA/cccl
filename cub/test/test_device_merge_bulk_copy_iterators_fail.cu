// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_merge.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

// Verifies that cub::DeviceMerge rejects a MergePolicy requesting bulk copy for keys if the two key ranges have
// different value types.

struct policy_selector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability) const -> cub::MergePolicy
  {
    return {128, 4, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE, true, false};
  }
};

int main()
{
  cuda::std::int64_t* keys1{};
  int* keys2{};
  // expected-error {{"MergePolicy: use_bulk_copy_for_keys requires both key ranges to be contiguous"}}
  [[maybe_unused]] const auto error = cub::DeviceMerge::MergeKeys(
    keys1, 0, keys2, 0, keys1, cuda::std::less<>{}, cuda::execution::tune(policy_selector{}));
}
