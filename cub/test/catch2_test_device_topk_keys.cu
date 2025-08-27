// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_topk.cuh>

#include <thrust/memory.h>
#include <thrust/sort.h>

#include <cuda/iterator>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKKeys, topk_keys);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKMinKeys, topk_min_keys);

using key_types       = c2h::type_list<cuda::std::uint16_t, float, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using k_items_types   = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
C2H_TEST("DeviceTopK::TopKKeys: Basic testing", "[keys][topk][device]", key_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = cuda::std::uint32_t;

  // Set input size
  constexpr num_items_t min_num_items = 1 << 2;
  constexpr num_items_t max_num_items = 1 << 15;
  const num_items_t num_items =
    GENERATE_COPY(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1 << 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, min(num_items - 1, max_k))));

  // Allocate the device memory
  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(k);

  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);
  c2h::host_vector<key_t> h_keys_in(keys_in);

  const bool is_descending = GENERATE(false, true);

  // Run the device-wide API
  if (!is_descending)
  {
    topk_min_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Sort the entire input data as result reference
  c2h::host_vector<key_t> host_results;
  host_results.resize(keys_out.size());
  if (is_descending)
  {
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), host_results.begin(), host_results.end(), std::greater<key_t>());
  }
  else
  {
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), host_results.begin(), host_results.end(), std::less<key_t>());
  }
  // Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  // We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  REQUIRE(host_results == device_results);
}

C2H_TEST("DeviceTopK::TopKKeys: works with iterators", "[keys][topk][device]", key_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = cuda::std::uint32_t;

  // Set input size
  constexpr num_items_t min_num_items = 1 << 2;
  constexpr num_items_t max_num_items = 1 << 24;
  const num_items_t num_items =
    GENERATE_COPY(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));
  // Large num_items will be in another test

  // Set the k value
  constexpr num_items_t min_k = 1 << 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, min(num_items - 1, max_k))));

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results

  /// Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  /// We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  if (is_descending)
  {
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_expected_it));
  }
  else
  {
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_in));
  }
}

C2H_TEST("DeviceTopK::TopKKeys: Test for large num_items", "[keys][topk][device]", num_items_types)
{
  using key_t       = uint32_t;
  using num_items_t = c2h::get<0, TestType>;

  // Set input size
  constexpr num_items_t max_num_items_ull =
    std::min(static_cast<size_t>(cuda::std::numeric_limits<num_items_t>::max()),
             cuda::std::numeric_limits<std::uint32_t>::max() + static_cast<size_t>(2000000ULL));
  constexpr num_items_t max_num_items = static_cast<num_items_t>(max_num_items_ull);
  const num_items_t num_items         = GENERATE_COPY(values({max_num_items}));

  // Set the k value
  constexpr num_items_t min_k = 1 << 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, min(num_items - 1, max_k))));

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results

  /// Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  /// We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  if (is_descending)
  {
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_expected_it));
  }
  else
  {
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_in));
  }
}

C2H_TEST("DeviceTopK::TopKKeys:  Test for different data types for num_items and k",
         "[keys][topk][device]",
         k_items_types)
{
  using key_t       = uint32_t;
  using num_items_t = uint64_t;
  using k_items_t   = c2h::get<0, TestType>;

  // Set input size
  constexpr num_items_t min_num_items = 1 << 2;
  constexpr num_items_t max_num_items = 1 << 15;
  const num_items_t num_items =
    GENERATE_COPY(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));
  // Large num_items will be in another test

  // Set the k value
  constexpr k_items_t min_k = 1 << 1;
  k_items_t limit_k         = min(cuda::std::numeric_limits<k_items_t>::max(), static_cast<k_items_t>(1 << 15));
  const k_items_t max_k =
    num_items - 1 < cuda::std::numeric_limits<k_items_t>::max()
      ? min(static_cast<k_items_t>(num_items - 1), limit_k)
      : limit_k;
  const k_items_t k = GENERATE_COPY(take(3, random(min_k, max_k)));

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results

  /// Since the results of API TopKMinKeys() and TopKKeys() are not-sorted
  /// We need to sort the results first.
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  if (is_descending)
  {
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_expected_it));
  }
  else
  {
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_in));
  }
}
