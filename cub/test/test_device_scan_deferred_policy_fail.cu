// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_scan.cuh>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

struct lookahead_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> cub::ScanPolicy
  {
    using selector_t =
      cub::detail::scan::policy_selector_from_types<int*, int*, int, cuda::std::uint32_t, cuda::std::plus<>, false, true>;
    auto policy      = selector_t{}(cc);
    policy.algorithm = cub::ScanAlgorithm::lookahead;
    policy.lookahead = cub::ScanLookaheadPolicy{4, 79, 3};
    return policy;
  }
};

int main()
{
  int* d_items     = nullptr;
  int* d_count     = nullptr;
  const auto count = cuda::args::deferred<int*>{d_count};
  const auto env   = cuda::execution::tune(lookahead_policy_selector{});
  // expected-error {{"Deferred DeviceScan counts require a tuning policy selecting lookback."}}
  const auto error = cub::DeviceScan::ExclusiveSum(d_items, d_items, count, env);
  return error == cudaSuccess ? 0 : 1;
}
