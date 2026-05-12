// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/functional>

#include "catch2_test_env_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:2

#include <c2h/catch2_test_helper.h>

// TODO(bgruber): the tests below should be refactored to call an env-overload that uses a memory resource to allocate
// temporary storage

namespace stdexec = cuda::std::execution;

auto topk_requirements()
{
  return cuda::execution::require(
    cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);
}

template <unsigned int ThreadsPerBlock>
struct topk_tuning
{
  _CCCL_API constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::detail::topk::topk_policy
  {
    return {ThreadsPerBlock, 1, 8, cub::BLOCK_LOAD_DIRECT, cub::BLOCK_SCAN_WARP_SCANS};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH != 1

C2H_TEST("DeviceTopK::MaxKeys can be tuned", "[topk][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size{0};

  block_size_extracting_constant_iterator input(42, thrust::raw_pointer_cast(d_block_size.data()));
  c2h::device_vector<int> d_keys_out(3);

  auto env = stdexec::env{topk_requirements(), cuda::execution::tune(topk_tuning<target_block_size>{})};

  size_t temp_size{};
  REQUIRE(cudaSuccess
          == cub::DeviceTopK::MaxKeys(
            nullptr, temp_size, input, d_keys_out.begin(), /* elements */ 1024, d_keys_out.size(), env));

  c2h::device_vector<char> temp(temp_size, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == cub::DeviceTopK::MaxKeys(
      thrust::raw_pointer_cast(temp.data()),
      temp_size,
      input,
      d_keys_out.begin(),
      /* elements */ 1024,
      d_keys_out.size(),
      env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceTopK::MinKeys can be tuned", "[topk][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size{0};

  block_size_extracting_constant_iterator input(42, thrust::raw_pointer_cast(d_block_size.data()));
  c2h::device_vector<int> d_keys_out(3);

  auto env = stdexec::env{topk_requirements(), cuda::execution::tune(topk_tuning<target_block_size>{})};

  size_t temp_size{};
  REQUIRE(cudaSuccess
          == cub::DeviceTopK::MinKeys(
            nullptr, temp_size, input, d_keys_out.begin(), /* elements */ 1024, d_keys_out.size(), env));

  c2h::device_vector<char> temp(temp_size, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == cub::DeviceTopK::MinKeys(
      thrust::raw_pointer_cast(temp.data()),
      temp_size,
      input,
      d_keys_out.begin(),
      /* elements */ 1024,
      d_keys_out.size(),
      env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceTopK::MaxPairs can be tuned", "[topk][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size{0};

  block_size_extracting_constant_iterator input(42, thrust::raw_pointer_cast(d_block_size.data()));
  auto values_in = cuda::make_counting_iterator<int>(0);
  c2h::device_vector<int> d_keys_out(3);
  c2h::device_vector<int> d_values_out(3);

  auto env = stdexec::env{topk_requirements(), cuda::execution::tune(topk_tuning<target_block_size>{})};

  size_t temp_size{};
  REQUIRE(cudaSuccess
          == cub::DeviceTopK::MaxPairs(
            nullptr, temp_size, input, d_keys_out.begin(), values_in, d_values_out.begin(), 1024, 3, env));

  c2h::device_vector<char> temp(temp_size, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == cub::DeviceTopK::MaxPairs(
      thrust::raw_pointer_cast(temp.data()),
      temp_size,
      input,
      d_keys_out.begin(),
      values_in,
      d_values_out.begin(),
      1024,
      3,
      env));
  REQUIRE(d_block_size[0] == target_block_size);
}

C2H_TEST("DeviceTopK::MinPairs can be tuned", "[topk][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size{0};

  block_size_extracting_constant_iterator input(42, thrust::raw_pointer_cast(d_block_size.data()));
  auto values_in = cuda::make_counting_iterator<int>(0);
  c2h::device_vector<int> d_keys_out(3);
  c2h::device_vector<int> d_values_out(3);

  auto env = stdexec::env{topk_requirements(), cuda::execution::tune(topk_tuning<target_block_size>{})};

  size_t temp_size{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceTopK::MinPairs(
      nullptr,
      temp_size,
      input,
      d_keys_out.begin(),
      values_in,
      d_values_out.begin(),
      /* elements */ 1024,
      d_keys_out.size(),
      env));

  c2h::device_vector<char> temp(temp_size, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == cub::DeviceTopK::MinPairs(
      thrust::raw_pointer_cast(temp.data()),
      temp_size,
      input,
      d_keys_out.begin(),
      values_in,
      d_values_out.begin(),
      1024,
      3,
      env));
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH != 1
