// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Define the public opt-out before any CCCL header so the whole translation unit sees dynamic cluster launches
// disabled.
#define CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH

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

// Verifies the public CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH opt-out. When defined, the automatic selector never picks the
// cluster backend, so the only way to reach the cluster arm is a `tune`d policy selector that forces it -- which the
// dispatch must reject at compile time (the kernel would otherwise launch without its runtime cluster extent). The
// request below is non-deterministic (a configuration the baseline backend serves), so the forced cluster backend comes
// solely from the override, not from the requirements -- keeping the test valid even if the baseline backend gains
// deterministic coverage in the future.

namespace
{
// Minimal tuning override that unconditionally forces the cluster backend, regardless of compute capability.
struct force_cluster_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::batched_topk::topk_policy
  {
    return cub::detail::batched_topk::topk_policy{
      cub::detail::batched_topk::topk_algorithm::cluster,
      cub::detail::batched_topk::make_baseline_policy(),
      cub::detail::batched_topk::make_cluster_policy()};
  }
};
} // namespace

int main()
{
  namespace ex = cuda::execution;

  int** d_keys_in    = nullptr;
  int** d_keys_out   = nullptr;
  auto segment_sizes = cuda::args::constant<8>{};
  auto k_arg         = cuda::args::constant<3>{};
  auto num_segments  = cuda::args::immediate{cuda::std::int64_t{2}};

  auto env = cuda::std::execution::env{
    ex::require(ex::determinism::not_guaranteed, ex::tie_break::unspecified, ex::output_ordering::unsorted),
    ex::tune(force_cluster_selector{})};
  // expected-error {{"a tuned policy selector forced the cluster backend"}}

  cuda::std::size_t temp_storage_bytes = 0;
  auto error                           = cub::DeviceBatchedTopK::MaxKeys(
    nullptr, temp_storage_bytes, d_keys_in, d_keys_out, segment_sizes, k_arg, num_segments, env);
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceBatchedTopK::MaxKeys failed with status: " << error << '\n';
  }
}
