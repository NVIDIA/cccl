// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/stream>

#include <c2h/catch2_test_helper.h>

#if _CCCL_STD_VER >= 2020

// example-begin reduce-by-key-policy-selector
struct ReduceByKeyPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability cc) const -> cub::ReduceByKeyPolicy
  {
    return {.threads_per_block = 128,
            .items_per_thread  = cc > cuda::compute_capability{9, 0} ? 7 : 6,
            .load_algorithm    = cub::BLOCK_LOAD_DIRECT,
            .load_modifier     = cub::LOAD_DEFAULT,
            .scan_algorithm    = cub::BLOCK_SCAN_WARP_SCANS,
            .lookback_delay    = cub::LookbackDelayPolicy{cub::LookbackDelayAlgorithm::fixed_delay, 832, 1165}};
  }
};
// example-end reduce-by-key-policy-selector

C2H_TEST("cub::DeviceReduce::ReduceByKey env-based API with tuning", "[reduce][env]")
{
  // example-begin reduce-by-key-tuning
  auto d_keys_in        = thrust::device_vector<int>{0, 2, 2, 9, 5, 5, 5, 8};
  auto d_values_in      = thrust::device_vector<int>{0, 7, 1, 6, 2, 5, 3, 4};
  auto d_unique_out     = thrust::device_vector<int>(8, thrust::no_init);
  auto d_aggregates_out = thrust::device_vector<int>(8, thrust::no_init);
  auto d_num_runs_out   = thrust::device_vector<int>(1, thrust::no_init);

  const auto error = cub::DeviceReduce::ReduceByKey(
    d_keys_in.begin(),
    d_unique_out.begin(),
    d_values_in.begin(),
    d_aggregates_out.begin(),
    d_num_runs_out.begin(),
    cuda::minimum<int>{},
    static_cast<int>(d_keys_in.size()),
    cuda::execution::tune(ReduceByKeyPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceReduce::ReduceByKey failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected_keys{0, 2, 9, 5, 8};
  thrust::device_vector<int> expected_aggregates{0, 1, 6, 2, 4};
  // example-end reduce-by-key-tuning

  REQUIRE(error == cudaSuccess);
  CHECK(d_num_runs_out[0] == 5);
  d_unique_out.resize(5);
  d_aggregates_out.resize(5);
  CHECK(d_unique_out == expected_keys);
  CHECK(d_aggregates_out == expected_aggregates);
}

#endif // _CCCL_STD_VER >= 2020
