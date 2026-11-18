// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <cuda/__functional/call_or.h>
#include <cuda/std/cstdint>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_custom_streams.cuh"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceFind::LowerBoundSortedValues, device_lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceFind::UpperBoundSortedValues, device_upper_bound_sorted_values);

DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::LowerBoundSortedValues, device_lower_bound_sorted_values_two_phase);
DECLARE_LAUNCH_WRAPPER(cub::DeviceFind::UpperBoundSortedValues, device_upper_bound_sorted_values_two_phase);

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

  const c2h::device_vector<int> expected = {0, 2, 2, 4};
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

  const c2h::device_vector<int> expected = {1, 2, 3, 4};
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

  const c2h::device_vector<int> expected = {0, 2, 2, 4};
  REQUIRE(d_output == expected);
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

  const c2h::device_vector<int> expected = {1, 2, 3, 4};
  REQUIRE(d_output == expected);
}

#if TEST_LAUNCH == 0

template <class EnvT, class TwoPhaseFn>
void run_two_phase(const EnvT& env, TwoPhaseFn two_phase)
{
  const stream_scope scope{cuda::__call_or(cuda::get_stream, cuda::stream_ref{cudaStream_t{}}, env).get()};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == two_phase(nullptr, temp_storage_bytes));

  c2h::device_vector<cuda::std::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  REQUIRE(cudaSuccess == two_phase(thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes));
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

// The launch wrappers copy their arguments, which the non-copyable streams of test_with_custom_streams do not allow
CUB_TEST_CASE("Device LowerBoundSortedValues two-phase works with custom streams",
              "[find][device][binary-search]",
              CUB_SMALL)
{
  const c2h::device_vector<int> d_range{0, 2, 4, 6, 8};
  const c2h::device_vector<int> d_values{0, 3, 4, 7};
  c2h::device_vector<int> d_output(4, thrust::no_init);
  const c2h::device_vector<int> expected{0, 2, 2, 4};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceFind::LowerBoundSortedValues(
        d_temp_storage,
        temp_storage_bytes,
        d_range.begin(),
        static_cast<int>(d_range.size()),
        d_values.begin(),
        static_cast<int>(d_values.size()),
        d_output.begin(),
        cuda::std::less{},
        env);
    });
    REQUIRE(d_output == expected);
  });
}

CUB_TEST_CASE("Device UpperBoundSortedValues two-phase works with custom streams",
              "[find][device][binary-search]",
              CUB_SMALL)
{
  const c2h::device_vector<int> d_range{0, 2, 4, 6, 8};
  const c2h::device_vector<int> d_values{0, 3, 4, 7};
  c2h::device_vector<int> d_output(4, thrust::no_init);
  const c2h::device_vector<int> expected{1, 2, 3, 4};

  test_with_custom_streams([&](const auto& env) {
    run_two_phase(env, [&](void* d_temp_storage, size_t& temp_storage_bytes) {
      return cub::DeviceFind::UpperBoundSortedValues(
        d_temp_storage,
        temp_storage_bytes,
        d_range.begin(),
        static_cast<int>(d_range.size()),
        d_values.begin(),
        static_cast<int>(d_values.size()),
        d_output.begin(),
        cuda::std::less{},
        env);
    });
    REQUIRE(d_output == expected);
  });
}

template <int ThreadsPerBlock>
struct find_bound_sorted_values_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::FindBoundSortedValuesPolicy
  {
    return {ThreadsPerBlock, 15, cub::LOAD_LDG};
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

CUB_TEST("Device LowerBoundSortedValues can be tuned", "[find][device][binary-search]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<int> d_range{0, 2, 4, 6, 8};
  const c2h::device_vector<int> d_values{0, 3, 4, 7};
  c2h::device_vector<int> d_output(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const block_size_extracting_op<cuda::std::less<>> comp{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(find_bound_sorted_values_tuning<target_block_size>{});

  SECTION("single-phase")
  {
    REQUIRE(
      cudaSuccess
      == cub::DeviceFind::LowerBoundSortedValues(
        d_range.begin(),
        static_cast<int>(d_range.size()),
        d_values.begin(),
        static_cast<int>(d_values.size()),
        d_output.begin(),
        comp,
        env));
  }
  SECTION("two-phase")
  {
    device_lower_bound_sorted_values_two_phase(
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      comp,
      env);
  }

  const c2h::device_vector<int> expected{0, 2, 2, 4};
  REQUIRE(d_output == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("Device UpperBoundSortedValues can be tuned", "[find][device][binary-search]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  const c2h::device_vector<int> d_range{0, 2, 4, 6, 8};
  const c2h::device_vector<int> d_values{0, 3, 4, 7};
  c2h::device_vector<int> d_output(4, thrust::no_init);
  c2h::device_vector<unsigned int> d_block_size(1, 0u);
  const block_size_extracting_op<cuda::std::less<>> comp{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(find_bound_sorted_values_tuning<target_block_size>{});

  SECTION("single-phase")
  {
    REQUIRE(
      cudaSuccess
      == cub::DeviceFind::UpperBoundSortedValues(
        d_range.begin(),
        static_cast<int>(d_range.size()),
        d_values.begin(),
        static_cast<int>(d_values.size()),
        d_output.begin(),
        comp,
        env));
  }
  SECTION("two-phase")
  {
    device_upper_bound_sorted_values_two_phase(
      d_range.begin(),
      static_cast<int>(d_range.size()),
      d_values.begin(),
      static_cast<int>(d_values.size()),
      d_output.begin(),
      comp,
      env);
  }

  const c2h::device_vector<int> expected{1, 2, 3, 4};
  REQUIRE(d_output == expected);
  REQUIRE(d_block_size[0] == target_block_size);
}

#endif // TEST_LAUNCH == 0

// The graph launch appends its own stream argument, which cannot be combined with an environment
#if TEST_LAUNCH != 2

// Before the environment parameter, the two-phase overloads took `cudaStream_t stream = nullptr`, so callers passing
// nullptr, a literal 0 or cudaStream_t{} for the stream exist. All must keep compiling and keep running on the default
// stream.
CUB_TEST_CASE("DeviceFind bound sorted values two-phase overloads accept legacy null stream arguments",
              "[find][device][binary-search]",
              CUB_SMALL)
{
  const c2h::device_vector<int> d_range{0, 2, 4, 6, 8};
  const c2h::device_vector<int> d_values{0, 3, 4, 7};
  c2h::device_vector<int> d_lower(4, thrust::no_init);
  c2h::device_vector<int> d_upper(4, thrust::no_init);
  const stream_scope scope{cudaStream_t{}};

  SECTION("nullptr")
  {
    device_lower_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_lower.begin(), cuda::std::less{}, nullptr);
    device_upper_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_upper.begin(), cuda::std::less{}, nullptr);
  }
  SECTION("literal 0")
  {
    device_lower_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_lower.begin(), cuda::std::less{}, 0);
    device_upper_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_upper.begin(), cuda::std::less{}, 0);
  }
  SECTION("cudaStream_t{}")
  {
    device_lower_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_lower.begin(), cuda::std::less{}, cudaStream_t{});
    device_upper_bound_sorted_values_two_phase(
      d_range.begin(), 5, d_values.begin(), 4, d_upper.begin(), cuda::std::less{}, cudaStream_t{});
  }

  REQUIRE(d_lower == c2h::device_vector<int>{0, 2, 2, 4});
  REQUIRE(d_upper == c2h::device_vector<int>{1, 2, 3, 4});
}

#endif // TEST_LAUNCH != 2

#if TEST_LAUNCH != 1
struct find_bound_tuning_transformer
{
  _CCCL_HOST_DEVICE_API constexpr auto
  operator()(cuda::compute_capability, cub::FindBoundSortedValuesPolicy policy) const
    -> cub::FindBoundSortedValuesPolicy
  {
    policy.threads_per_block = 64;
    return policy;
  }
};

template <typename CompT, typename EnvT>
unsigned int lower_bound_block_size(
  c2h::device_vector<int>& d_range,
  c2h::device_vector<int>& d_values,
  c2h::device_vector<int>& d_output,
  CompT comp,
  c2h::device_vector<unsigned int>& d_block_size,
  EnvT env)
{
  d_block_size[0] = 0;
  device_lower_bound_sorted_values(
    d_range.begin(),
    static_cast<int>(d_range.size()),
    d_values.begin(),
    static_cast<int>(d_values.size()),
    d_output.begin(),
    comp,
    env);
  return d_block_size[0];
}

CUB_TEST("Device LowerBoundSortedValues tuning can be modified", "[find][device][binary-search]", CUB_SMALL)
{
  constexpr unsigned int target_block_size = 64; // must be different from any value we use in production

  auto d_range      = c2h::device_vector<int>{0, 2, 4, 6, 8};
  auto d_values     = c2h::device_vector<int>{0, 3, 4, 7};
  auto d_output     = c2h::device_vector<int>(4, thrust::no_init);
  auto d_block_size = c2h::device_vector<unsigned int>(1, 0);

  const block_size_extracting_op<cuda::std::less<>> comp{thrust::raw_pointer_cast(d_block_size.data())};

  const c2h::device_vector<int> expected = {0, 2, 2, 4};

  {
    // CUB's default tuning must not use the block size we are going to request
    const unsigned int default_block_size =
      lower_bound_block_size(d_range, d_values, d_output, comp, d_block_size, stdexec::env<>{});
    REQUIRE(d_output == expected);
    REQUIRE(default_block_size != 0);
    REQUIRE(default_block_size != target_block_size);
  }

  REQUIRE(lower_bound_block_size(
            d_range, d_values, d_output, comp, d_block_size, cuda::execution::tune(find_bound_tuning_transformer{}))
          == target_block_size);
  REQUIRE(d_output == expected);
}
#endif // TEST_LAUNCH != 1

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
