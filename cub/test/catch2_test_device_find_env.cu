// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/execution>

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
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::FindPolicy
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

  SECTION("Without provided memory")
  {
    auto error = cub::DeviceFind::FindIf(d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_out[0] == 5);
  }

  SECTION("With provided memory")
  {
    void* temp_storage = nullptr;
    size_t num_bytes   = 0;
    auto error = cub::DeviceFind::FindIf(temp_storage, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp  = c2h::device_vector<uint8_t>(num_bytes, thrust::no_init);
    temp_storage = thrust::raw_pointer_cast(d_temp.data());

    error = cub::DeviceFind::FindIf(temp_storage, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_out[0] == 5);
  }
}

TEST_CASE("Device FindIf no match returns num_items with default environment", "[find][device]")
{
  constexpr int num_items = 5;
  auto d_in               = c2h::device_vector<int>{0, 1, 2, 3, 4};
  auto d_out              = c2h::device_vector<int>(1);
  is_greater_than_t predicate{100};

  SECTION("Without provided memory")
  {
    auto error = cub::DeviceFind::FindIf(d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_out[0] == num_items);
  }

  SECTION("With provided memory")
  {
    void* temp_storage = nullptr;
    size_t num_bytes   = 0;
    auto error = cub::DeviceFind::FindIf(temp_storage, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    auto d_temp  = c2h::device_vector<uint8_t>(num_bytes, thrust::no_init);
    temp_storage = thrust::raw_pointer_cast(d_temp.data());

    error = cub::DeviceFind::FindIf(temp_storage, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_out[0] == num_items);
  }
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

C2H_TEST("Device FindIf works with user provided memory and environment", "[find][device]")
{
  constexpr int num_items = 8;
  auto d_in               = c2h::device_vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
  auto d_out              = c2h::device_vector<int>(1);
  is_greater_than_t predicate{4};

  size_t expected_bytes_allocated{};
  auto error =
    cub::DeviceFind::FindIf(nullptr, expected_bytes_allocated, d_in.begin(), d_out.begin(), predicate, num_items);
  REQUIRE(error == cudaSuccess);
  auto temp          = c2h::device_vector<uint8_t>(expected_bytes_allocated, thrust::no_init);
  void* temp_storage = thrust::raw_pointer_cast(temp.data());

  auto test_find_if = [&](const auto& env) {
    size_t num_bytes = 0;
    error = cub::DeviceFind::FindIf(nullptr, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items, env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(num_bytes == expected_bytes_allocated);

    error = cub::DeviceFind::FindIf(temp_storage, num_bytes, d_in.begin(), d_out.begin(), predicate, num_items, env);
    REQUIRE(error == cudaSuccess);
    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());
    REQUIRE(d_out[0] == 5);
  };

  int current_device;
  error = cudaGetDevice(&current_device);
  REQUIRE(error == cudaSuccess);

  SECTION("find_if works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_find_if(stream.get());
  }

  SECTION("find_if works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_find_if(stream);
  }

  SECTION("find_if works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_find_if(stream_ref);
  }

  SECTION("find_if works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_find_if(env);
  }

  SECTION("find_if works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_find_if(policy);
  }

  SECTION("find_if works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_find_if(policy);
  }
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

C2H_TEST("Device LowerBound works with user provided memory and environment", "[find][device]")
{
  auto d_range                     = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values                    = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output                    = c2h::device_vector<int>(4);
  c2h::device_vector<int> expected = {1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceFind::LowerBound(
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

  auto test_lower_bound = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceFind::LowerBound(
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

    error = cub::DeviceFind::LowerBound(
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

  SECTION("lower_bound works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_lower_bound(stream.get());
  }

  SECTION("lower_bound works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_lower_bound(stream);
  }

  SECTION("lower_bound works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_lower_bound(stream_ref);
  }

  SECTION("lower_bound works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_lower_bound(env);
  }

  SECTION("lower_bound works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_lower_bound(policy);
  }

  SECTION("lower_bound works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_lower_bound(policy);
  }
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

C2H_TEST("Device UpperBound works with user provided memory and environment", "[find][device]")
{
  auto d_range                     = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values                    = c2h::device_vector<int>{1, 3, 5, 7};
  auto d_output                    = c2h::device_vector<int>(4);
  c2h::device_vector<int> expected = {1, 2, 3, 4};

  size_t expected_bytes_allocated{};
  auto error = cub::DeviceFind::UpperBound(
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

  auto test_upper_bound = [&](const auto& env) {
    size_t num_bytes = 0;
    error            = cub::DeviceFind::UpperBound(
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

    error = cub::DeviceFind::UpperBound(
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

  SECTION("upper_bound works with cudaStream_t")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_upper_bound(stream.get());
  }

  SECTION("upper_bound works with cuda::stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    test_upper_bound(stream);
  }

  SECTION("upper_bound works with cuda::stream_ref")
  {
    cuda::stream stream{cuda::devices[current_device]};
    cuda::stream_ref stream_ref{stream};
    test_upper_bound(stream_ref);
  }

  SECTION("upper_bound works with cuda::std::execution::env")
  {
    cuda::std::execution::env env{};
    test_upper_bound(env);
  }

  SECTION("upper_bound works with cuda::execution::gpu")
  {
    const auto policy = cuda::execution::gpu;
    test_upper_bound(policy);
  }

  SECTION("upper_bound works with cuda::execution::gpu with stream")
  {
    cuda::stream stream{cuda::devices[current_device]};
    const auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);
    test_upper_bound(policy);
  }
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

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
C2H_TEST("FindPolicy", "[find][device]")
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::FindPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::FindPolicy>);

  // aggregate init
  constexpr auto p1 = cub::FindPolicy{128, 7, 4, cub::CacheLoadModifier::LOAD_LDG};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::FindPolicy{
    .threads_per_block = 128, .items_per_thread = 7, .vec_size = 4, .load_modifier = cub::CacheLoadModifier::LOAD_LDG};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
