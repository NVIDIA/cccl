// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/functional>
#include <cuda/iterator>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::FindIf, device_find_if);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBound, device_lower_bound);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBound, device_upper_bound);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include <c2h/catch2_test_helper.h>

namespace stdexec = cuda::std::execution;

struct is_greater_than_t
{
  int threshold;
  __host__ __device__ bool operator()(int value) const
  {
    return value > threshold;
  }
};

// A policy selector that forces a specific block size, so a test can verify the tuning was applied.
template <int ThreadsPerBlock>
struct find_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::detail::find::find_policy
  {
    return {ThreadsPerBlock, 4, 4, cub::LOAD_LDG};
  }
};

using block_size_extracting_predicate_t = block_size_extracting_op<::cuda::always_false>;

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

#if TEST_LAUNCH == 0

TEST_CASE("Device FindIf works with default environment", "[find][device]")
{
  constexpr int num_items = 8;
  auto d_in               = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto d_out              = c2h::device_vector<int>(1);
  is_greater_than_t predicate{4};

  auto error = cub::DeviceFind::FindIf(d_in.begin(), d_out.begin(), predicate, num_items);
  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out[0] == 5);
}

TEST_CASE("Device FindIf no match returns num_items with default environment", "[find][device]")
{
  constexpr int num_items = 5;
  auto d_in               = c2h::device_vector<int>{0, 1, 2, 3, 4};
  auto d_out              = c2h::device_vector<int>(1);
  is_greater_than_t predicate{100};

  auto error = cub::DeviceFind::FindIf(d_in.begin(), d_out.begin(), predicate, num_items);
  REQUIRE(error == cudaSuccess);
  REQUIRE(d_out[0] == num_items);
}

TEST_CASE("Device LowerBound works with default environment", "[find][device]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::LowerBound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

TEST_CASE("Device UpperBound works with default environment", "[find][device]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::UpperBound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

#endif

C2H_TEST("Device FindIf uses environment", "[find][device]")
{
  constexpr int num_items = 8;
  auto d_in               = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto d_out              = c2h::device_vector<int>(1);
  is_greater_than_t predicate{4};

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::FindIf(nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), predicate, num_items));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_find_if(d_in.begin(), d_out.begin(), predicate, num_items, env);

  REQUIRE(d_out[0] == 5);
}

C2H_TEST("Device LowerBound uses environment", "[find][device]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::LowerBound(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_lower_bound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    env);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

C2H_TEST("Device UpperBound uses environment", "[find][device]")
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::UpperBound(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_upper_bound(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    env);

  c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

#if TEST_LAUNCH != 1
C2H_TEST("Device FindIf can be tuned", "[find][device]", block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;

  constexpr int num_items = 1024;
  auto d_in               = c2h::device_vector<int>(num_items, 0);
  auto d_out              = c2h::device_vector<int>(1, thrust::no_init);
  auto d_block_size       = c2h::device_vector<unsigned int>(1, 0);

  block_size_extracting_predicate_t predicate{thrust::raw_pointer_cast(d_block_size.data())};

  auto env = cuda::execution::tune(find_tuning<static_cast<int>(target_block_size)>{});

  device_find_if(d_in.begin(), d_out.begin(), predicate, num_items, env);

  REQUIRE(d_out[0] == num_items); // predicate never matches
  REQUIRE(d_block_size[0] == target_block_size);
}
#endif // TEST_LAUNCH != 1
