// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_find.cuh>

#include <thrust/device_vector.h>

#include <sstream>

#include "block_size_extracting_helpers.h"
#include "catch2_test_launch_helper.h"

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceFind::LowerBoundSortedValues, device_lower_bound_sorted_values);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceFind::UpperBoundSortedValues, device_upper_bound_sorted_values);

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
  auto d_output     = c2h::device_vector<int>(4);
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
