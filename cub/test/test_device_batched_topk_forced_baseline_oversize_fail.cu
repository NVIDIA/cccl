// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/execution>

#include <iostream>

// Verifies the strict forced-baseline-oversize diagnostic in cub::DeviceBatchedTopK (the static_assert in
// launch_baseline_arm, cub/device/dispatch/dispatch_batched_topk.cuh). The automatic selector never routes an oversize
// segment to the baseline backend; only a trusted `tune`d override can. Here an explicitly non-deterministic request
// (so it takes the baseline arm, not the deterministic-forced-baseline path) forces the baseline backend with the
// default policy -- largest worker tile 16384 keys -- against a 2^20-key static maximum segment size that no baseline
// worker policy covers. In strict (default) mode this must fail to compile rather than defer to a runtime
// cudaErrorNotSupported; built *without* CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT to exercise that path. A
// forced-baseline selector never resolves to `unsupported`, so the arch-unsupported static_assert does not pre-empt it.
struct forced_baseline_oversize_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::batched_topk::topk_policy
  {
    return cub::detail::batched_topk::topk_policy{
      cub::detail::batched_topk::topk_algorithm::baseline,
      cub::detail::batched_topk::make_baseline_policy(),
      cub::detail::batched_topk::make_cluster_policy()};
  }
};

int main()
{
  namespace ex = cuda::execution;

  int** d_keys_in    = nullptr;
  int** d_keys_out   = nullptr;
  auto segment_sizes = cuda::args::constant<(1 << 20)>{};
  auto k_arg         = cuda::args::constant<3>{};
  auto num_segments  = cuda::args::immediate{cuda::std::int64_t{2}};

  auto requirements =
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::unspecified, ex::output_ordering::unsorted);
  auto env = cuda::std::execution::env{requirements, cuda::execution::tune(forced_baseline_oversize_selector{})};
  // expected-error {{"cannot cover the static maximum segment size"}}

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
