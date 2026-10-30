// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_merge.cuh>

#include <cuda/__execution/tune.h>
#include <cuda/std/array>
#include <cuda/std/functional>

// Verifies that cub::DeviceMerge rejects a MergePolicy requesting bulk copy for keys whose size differs from their
// alignment.

struct policy_selector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability) const -> cub::MergePolicy
  {
    return {128, 4, cub::LOAD_DEFAULT, cub::BLOCK_STORE_WARP_TRANSPOSE, true, false};
  }
};

int main()
{
  cuda::std::array<int, 3>* keys{}; // sizeof == 12, alignof == 4
  // expected-error {{"MergePolicy: use_bulk_copy_for_keys requires a key type whose size equals its alignment"}}
  [[maybe_unused]] const auto error =
    cub::DeviceMerge::MergeKeys(keys, 0, keys, 0, keys, cuda::std::less<>{}, cuda::execution::tune(policy_selector{}));
}
