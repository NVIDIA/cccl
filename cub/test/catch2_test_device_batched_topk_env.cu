// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_batched_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>
#include <cuda/std/functional>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceBatchedTopK::MaxKeys, device_batched_topk_max_keys);

// %PARAM% TEST_LAUNCH lid 0:2

#include "cub_test_macros.h"

#if TEST_LAUNCH != 1

struct batched_topk_tuning_transformer
{
  _CCCL_HOST_DEVICE_API constexpr auto
  operator()(cuda::compute_capability, cub::detail::batched_topk::topk_policy policy) const
    -> cub::detail::batched_topk::topk_policy
  {
    policy.backend = cub::detail::batched_topk::topk_algorithm::baseline;
    for (auto& worker : policy.baseline.worker_per_segment_policies.elems)
    {
      worker.threads_per_block = 64;
    }
    policy.baseline.multi_worker_per_segment_policy.threads_per_block = 64;
    return policy;
  }
};

template <typename EnvT>
unsigned int batched_topk_block_size(
  thrust::device_vector<int>& keys_out, thrust::device_vector<unsigned int>& d_block_size, EnvT env)
{
  constexpr int num_segments                  = 2;
  [[maybe_unused]] constexpr int segment_size = 8; // MSVC does not count template arguments as uses
  constexpr int k                             = 3;

  d_block_size[0] = 0;
  // all segments read the same keys, which record the block size of the kernel reading them
  auto d_keys_in = cuda::make_constant_iterator(
    block_size_extracting_constant_iterator{42, thrust::raw_pointer_cast(d_block_size.data())});
  auto d_keys_out =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(keys_out.data())), k);

  device_batched_topk_max_keys(
    d_keys_in,
    d_keys_out,
    cuda::args::constant<segment_size>{},
    cuda::args::constant<k>{},
    cuda::args::immediate{cuda::std::int64_t{num_segments}},
    env);
  return d_block_size[0];
}

CUB_TEST("cub::DeviceBatchedTopK::MaxKeys tuning can be modified", "[batched_topk][device][env]", CUB_SMALL)
{
  constexpr int k                          = 3;
  constexpr unsigned int target_block_size = 64;

  auto keys_out     = thrust::device_vector<int>(2 * k, thrust::no_init);
  auto d_block_size = thrust::device_vector<unsigned int>(1, 0);

  auto requirements = cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::tie_break::unspecified,
    cuda::execution::output_ordering::unsorted);

  {
    // CUB's default tuning must not use the block size we are going to request
    const unsigned int default_block_size =
      batched_topk_block_size(keys_out, d_block_size, cuda::std::execution::env{requirements});
    REQUIRE(default_block_size != 0);
    REQUIRE(default_block_size != target_block_size);
  }

  REQUIRE(batched_topk_block_size(
            keys_out,
            d_block_size,
            cuda::std::execution::env{requirements, cuda::execution::tune(batched_topk_tuning_transformer{})})
          == target_block_size);
  REQUIRE(keys_out == thrust::device_vector<int>(2 * k, 42));
}

#endif // TEST_LAUNCH != 1
