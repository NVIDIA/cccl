// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_radix_sort.cuh>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/std/cstdint>

struct multi_pass_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const
    -> cub::RadixSortPolicy
  {
    auto policy =
      cub::detail::radix_sort::policy_selector_from_types<int, cub::NullType, cuda::std::uint32_t, true>{}(cc);
    policy.algorithm = cub::RadixSortAlgorithm::multi_pass;
    return policy;
  }
};

int main()
{
  cub::DoubleBuffer<int> keys;
  const auto count = cuda::args::deferred<int*>{nullptr};
  const auto env   = cuda::execution::tune(multi_pass_policy_selector{});
  // expected-error {{"Deferred DeviceRadixSort counts require a tuning policy selecting onesweep."}}
  const auto error = cub::DeviceRadixSort::SortKeys(keys, count, 0, 32, env);
  return error == cudaSuccess ? 0 : 1;
}
