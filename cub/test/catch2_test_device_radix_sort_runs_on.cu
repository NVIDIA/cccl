// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Should precede any includes
struct stream_registry_factory_t;
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY stream_registry_factory_t

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/util_device.cuh>

#include <cuda/execution.runs_on.h>
#include <cuda/execution>

#include <cstdint>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_launch_helper.h"
#include "cub_test_macros.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeys, device_radix_sort_keys);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortKeysDescending, device_radix_sort_keys_descending);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairs, device_radix_sort_pairs);
DECLARE_LAUNCH_WRAPPER_ENV(cub::DeviceRadixSort::SortPairsDescending, device_radix_sort_pairs_descending);

CUB_TEST("Device radix sort keys with runs_on sorts correctly", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto env = cuda::execution::guarantee(cuda::execution::experimental::runs_on{cc});

  c2h::device_vector<std::int32_t> keys_in(num_items);
  c2h::device_vector<std::int32_t> keys_out(num_items);
  c2h::gen(c2h::seed_t{42}, keys_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, false);
    device_radix_sort_keys(keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, env);
    REQUIRE(keys_out == expected);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, true);
    device_radix_sort_keys_descending(keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, env);
    REQUIRE(keys_out == expected);
  }
}

CUB_TEST("Device radix sort keys with runs_on handles an SM limit", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  cuda::execution::experimental::device_description description{};
  description.__max_sms_ = 1;
  const auto env         = cuda::execution::guarantee(cuda::execution::experimental::runs_on{cc, description});

  c2h::device_vector<std::int32_t> keys_in(num_items);
  c2h::device_vector<std::int32_t> keys_out(num_items);
  c2h::gen(c2h::seed_t{42}, keys_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, false);
    device_radix_sort_keys(keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, env);
    REQUIRE(keys_out == expected);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, true);
    device_radix_sort_keys_descending(keys_in.data().get(), keys_out.data().get(), num_items, 0, 32, env);
    REQUIRE(keys_out == expected);
  }
}

CUB_TEST("Device radix sort pairs with runs_on sorts correctly", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto env = cuda::execution::guarantee(cuda::execution::experimental::runs_on{cc});

  c2h::device_vector<std::int32_t> keys_in(num_items);
  c2h::device_vector<std::int32_t> keys_out(num_items);
  c2h::device_vector<std::int32_t> values_in(num_items);
  c2h::device_vector<std::int32_t> values_out(num_items);
  c2h::gen(c2h::seed_t{42}, keys_in, std::int32_t{-10}, std::int32_t{10});
  c2h::gen(c2h::seed_t{123}, values_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, false);
    device_radix_sort_pairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      num_items,
      0,
      32,
      env);
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, true);
    device_radix_sort_pairs_descending(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      num_items,
      0,
      32,
      env);
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }
}

CUB_TEST("Device radix sort pairs with runs_on handles an SM limit", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  const auto num_items = GENERATE(0, 1, 42, 100'000);
  CAPTURE(num_items);
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  cuda::execution::experimental::device_description description{};
  description.__max_sms_ = 1;
  const auto env         = cuda::execution::guarantee(cuda::execution::experimental::runs_on{cc, description});

  c2h::device_vector<std::int32_t> keys_in(num_items);
  c2h::device_vector<std::int32_t> keys_out(num_items);
  c2h::device_vector<std::int32_t> values_in(num_items);
  c2h::device_vector<std::int32_t> values_out(num_items);
  c2h::gen(c2h::seed_t{42}, keys_in, std::int32_t{-10}, std::int32_t{10});
  c2h::gen(c2h::seed_t{123}, values_in);

  SECTION("Ascending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, false);
    device_radix_sort_pairs(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      num_items,
      0,
      32,
      env);
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }

  SECTION("Descending")
  {
    const auto expected = radix_sort_reference(keys_in, values_in, true);
    device_radix_sort_pairs_descending(
      keys_in.data().get(),
      keys_out.data().get(),
      values_in.data().get(),
      values_out.data().get(),
      num_items,
      0,
      32,
      env);
    REQUIRE(keys_out == expected.first);
    REQUIRE(values_out == expected.second);
  }
}

#if TEST_LAUNCH == 0 && defined(CCCL_ENABLE_ASSERTIONS)
CUB_TEST("Device radix sort rejects a mismatched runs_on capability", "[radix_sort][device][runs_on]", CUB_SMALL)
{
  cuda::compute_capability cc{};
  REQUIRE(cudaSuccess == cub::detail::ptx_compute_cap(cc));
  const auto wrong_cc =
    cc == cuda::compute_capability{8, 0} ? cuda::compute_capability{9, 0} : cuda::compute_capability{8, 0};
  const auto env = cuda::execution::guarantee(cuda::execution::experimental::runs_on{wrong_cc});
  c2h::device_vector<std::int32_t> keys_in{3, 1, 2};
  c2h::device_vector<std::int32_t> keys_out(3);

  REQUIRE(cudaErrorInvalidDevice
          == cub::DeviceRadixSort::SortKeys(keys_in.data().get(), keys_out.data().get(), 3, 0, 32, env));
}
#endif // TEST_LAUNCH == 0 && defined(CCCL_ENABLE_ASSERTIONS)
