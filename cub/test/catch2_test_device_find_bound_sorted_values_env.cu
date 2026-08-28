// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/execution>

#include <sstream>

#include "catch2_test_env_launch_helper.h"

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBoundSortedValues, device_lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBoundSortedValues, device_upper_bound_sorted_values);

// %PARAM% TEST_LAUNCH lid 0:1:2

#include "cub_test_macros.h"

namespace stdexec = cuda::std::execution;

#if TEST_LAUNCH == 0

CUB_TEST_CASE("Device LowerBoundSortedValues works with default environment", "[find][device][binary-search]", CUB_SMALL)
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::LowerBoundSortedValues(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);

  c2h::device_vector<int> expected = {0, 2, 2, 4};
  REQUIRE(d_output == expected);
}

CUB_TEST_CASE("Device UpperBoundSortedValues works with default environment", "[find][device][binary-search]", CUB_SMALL)
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  auto error = cub::DeviceFind::UpperBoundSortedValues(
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

CUB_TEST("Device LowerBoundSortedValues uses environment", "[find][device][binary-search]", CUB_SMALL)
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::LowerBoundSortedValues(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_lower_bound_sorted_values(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{},
    env);

  c2h::device_vector<int> expected = {0, 2, 2, 4};
  REQUIRE(d_output == expected);
}

CUB_TEST("Device LowerBoundSortedValues works with user provided memory and environment",
         "[find][device][binary-search]",
         CUB_SMALL)
{
  auto d_range                     = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values                    = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output                    = c2h::device_vector<int>(4);
  c2h::device_vector<int> expected = {0, 2, 2, 4};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceFind::LowerBoundSortedValues(
    nullptr,
    expected_bytes_allocated,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  auto temp          = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(temp.data());

  auto test_lower_bound_sorted_values = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceFind::LowerBoundSortedValues(
      nullptr,
      num_bytes,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{},
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(num_bytes == expected_bytes_allocated);

    error = cub::DeviceFind::LowerBoundSortedValues(
      temp_storage,
      num_bytes,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{},
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_output == expected);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("lower_bound_sorted_values works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_lower_bound_sorted_values(stream.get());
  }

  SECTION("lower_bound_sorted_values works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_lower_bound_sorted_values(stream);
  }

  SECTION("lower_bound_sorted_values works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_lower_bound_sorted_values(stream_ref);
  }

  SECTION("lower_bound_sorted_values works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_lower_bound_sorted_values(env);
  }

  SECTION("lower_bound_sorted_values works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_lower_bound_sorted_values(policy);
  }

  SECTION("lower_bound_sorted_values works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_lower_bound_sorted_values(policy);
  }
}

CUB_TEST("Device UpperBoundSortedValues uses environment", "[find][device][binary-search]", CUB_SMALL)
{
  auto d_range  = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output = c2h::device_vector<int>(4);

  size_t expected_bytes_allocated{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceFind::UpperBoundSortedValues(
      nullptr,
      expected_bytes_allocated,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{}));

  auto env = stdexec::env{expected_allocation_size(expected_bytes_allocated)};

  device_upper_bound_sorted_values(
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

CUB_TEST("Device UpperBoundSortedValues works with user provided memory and environment",
         "[find][device][binary-search]",
         CUB_SMALL)
{
  auto d_range                     = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values                    = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output                    = c2h::device_vector<int>(4);
  c2h::device_vector<int> expected = {1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceFind::UpperBoundSortedValues(
    nullptr,
    expected_bytes_allocated,
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    cuda::std::less{});
  REQUIRE(error == cudaSuccess);
  auto temp          = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(temp.data());

  auto test_upper_bound_sorted_values = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceFind::UpperBoundSortedValues(
      nullptr,
      num_bytes,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{},
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(num_bytes == expected_bytes_allocated);

    error = cub::DeviceFind::UpperBoundSortedValues(
      temp_storage,
      num_bytes,
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      cuda::std::less{},
      env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_output == expected);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("upper_bound_sorted_values works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_upper_bound_sorted_values(stream.get());
  }

  SECTION("upper_bound_sorted_values works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_upper_bound_sorted_values(stream);
  }

  SECTION("upper_bound_sorted_values works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_upper_bound_sorted_values(stream_ref);
  }

  SECTION("upper_bound_sorted_values works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_upper_bound_sorted_values(env);
  }

  SECTION("upper_bound_sorted_values works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_upper_bound_sorted_values(policy);
  }

  SECTION("upper_bound_sorted_values works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_upper_bound_sorted_values(policy);
  }
}

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test FindBoundSortedValuesPolicy properties", "[find][device][binary-search]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::FindBoundSortedValuesPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::FindBoundSortedValuesPolicy>);

  // aggregate init
  constexpr auto p1 = cub::FindBoundSortedValuesPolicy{256, 15, cub::CacheLoadModifier::LOAD_LDG};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::FindBoundSortedValuesPolicy{
    .threads_per_block = 256, .items_per_thread = 15, .load_modifier = cub::CacheLoadModifier::LOAD_LDG};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1)
          == "FindBoundSortedValuesPolicy { .threads_per_block = 256, .items_per_thread = 15"
             ", .load_modifier = LOAD_LDG }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
