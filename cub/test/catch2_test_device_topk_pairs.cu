// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_topk.cuh>

#include <cuda/iterator>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKPairs, topk_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceTopK::TopKMinPairs, topk_min_pairs);
DECLARE_LAUNCH_WRAPPER(cub::DeviceMergeSort::StableSortKeys, stable_sort_keys);

template <typename key_t, typename value_t, typename num_items_t>
void sort_keys_and_values(
  c2h::device_vector<key_t>& keys, c2h::device_vector<value_t>& values, num_items_t num_items, bool is_descending)
{
  auto zipped_it = cuda::make_zip_iterator(keys.begin(), values.begin());

  // Perform sort
  if (is_descending)
  {
    stable_sort_keys(zipped_it, num_items, cuda::std::greater<>{});
  }
  else
  {
    stable_sort_keys(zipped_it, num_items, cuda::std::less<>{});
  }
}

template <typename key_in_it, typename value_in_it, typename key_t, typename value_t, typename num_items_t>
bool check_results(
  key_in_it h_keys_in,
  value_in_it h_values_in,
  c2h::device_vector<key_t>& keys_out,
  c2h::device_vector<value_t>& values_out,
  num_items_t num_items,
  num_items_t k,
  bool is_descending)
{
  // Since the results of API TopKMinPairs() and TopKPairs() are not-sorted
  // We need to sort the results first.
  sort_keys_and_values(keys_out, values_out, k, is_descending);
  c2h::host_vector<key_t> h_keys_out(keys_out);
  c2h::host_vector<value_t> h_values_out(values_out);

  // i for results from gpu (TopKMinPairs() and TopKPairs()); j for reference results
  num_items_t i = 0, j = 0;
  bool res = true;
  while (i < k && j < num_items)
  {
    if (h_keys_out[i] == h_keys_in[j])
    {
      if (h_values_out[i] == h_values_in[j])
      {
        i++;
        j++;
      }
      else if (is_descending ? h_values_out[i] < h_values_in[j] : h_values_out[i] > h_values_in[j])
      {
        // Note: The results returned by the API functions TopKMinPairs() and TopKPairs() are not stable.
        // If there are multiple items whose keys are equal to the key of the k-th element, any of those items may
        // appear in the results. Therefore, when the value does not match, we increment 'j' to continue searching for a
        // matching value with the same key.
        j++;
      }
      else
      {
        res = false;
        break;
      }
    }
    else
    {
      res = false;
      break;
    }
  }
  return res;
}

template <typename key_in_it, typename value_in_it, typename key_t, typename value_t, typename num_items_t, typename k_items_t>
bool topk_with_iterator(
  key_in_it keys_in,
  value_in_it values_in,
  c2h::device_vector<key_t>& keys_out,
  c2h::device_vector<value_t>& values_out,
  num_items_t num_items,
  k_items_t k,
  bool is_descending)
{
  if (!is_descending)
  {
    topk_min_pairs(
      keys_in,
      thrust::raw_pointer_cast(keys_out.data()),
      values_in,
      thrust::raw_pointer_cast(values_out.data()),
      num_items,
      k);
  }
  else
  {
    topk_pairs(keys_in,
               thrust::raw_pointer_cast(keys_out.data()),
               values_in,
               thrust::raw_pointer_cast(values_out.data()),
               num_items,
               k);
  }

  // Verify results
  bool res;
  if (is_descending)
  {
    auto keys_expected_it   = cuda::std::make_reverse_iterator(keys_in + num_items);
    auto values_expected_it = cuda::std::make_reverse_iterator(values_in + num_items);
    res                     = check_results(
      keys_expected_it, values_expected_it, keys_out, values_out, num_items, static_cast<num_items_t>(k), is_descending);
  }
  else
  {
    auto keys_expected_it   = keys_in;
    auto values_expected_it = values_in;
    res                     = check_results(
      keys_expected_it, values_expected_it, keys_out, values_out, num_items, static_cast<num_items_t>(k), is_descending);
  }
  return res;
}

using key_types       = c2h::type_list<cuda::std::uint16_t, float, cuda::std::uint64_t>;
using value_types     = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using k_items_types   = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

C2H_TEST("DeviceTopK::TopKPairs: Basic testing", "[pairs][topk][device]", key_types, value_types)
{
  using key_t       = c2h::get<0, TestType>;
  using value_t     = c2h::get<1, TestType>;
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

  c2h::device_vector<value_t> values_in(num_items);
  c2h::device_vector<value_t> values_out(k);

  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);
  c2h::gen(C2H_SEED(num_value_seeds), values_in);

  const bool select_min    = GENERATE(false, true);
  const bool is_descending = !select_min;

  // Run the device-wide API
  if (select_min)
  {
    topk_min_pairs(
      thrust::raw_pointer_cast(keys_in.data()),
      thrust::raw_pointer_cast(keys_out.data()),
      thrust::raw_pointer_cast(values_in.data()),
      thrust::raw_pointer_cast(values_out.data()),
      num_items,
      k);
  }
  else
  {
    topk_pairs(thrust::raw_pointer_cast(keys_in.data()),
               thrust::raw_pointer_cast(keys_out.data()),
               thrust::raw_pointer_cast(values_in.data()),
               thrust::raw_pointer_cast(values_out.data()),
               num_items,
               k);
  }

  // Sort the entire input data as result reference
  sort_keys_and_values(keys_in, values_in, num_items, is_descending);
  c2h::host_vector<key_t> h_keys(keys_in);
  c2h::host_vector<value_t> h_values(values_in);

  bool res = check_results(
    thrust::raw_pointer_cast(h_keys.data()),
    thrust::raw_pointer_cast(h_values.data()),
    keys_out,
    values_out,
    num_items,
    k,
    is_descending);

  REQUIRE(res == true);
}

C2H_TEST("DeviceTopK::TopKPairs: Works with iterators", "[pairs][topk][device]", key_types, value_types, num_items_types)
{
  using key_t       = c2h::get<0, TestType>;
  using value_t     = c2h::get<1, TestType>;
  using num_items_t = c2h::get<2, TestType>;

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
  auto keys_in   = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  auto values_in = cuda::make_counting_iterator(value_t{});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));
  c2h::device_vector<value_t> values_out(k, static_cast<value_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  bool res                 = topk_with_iterator(keys_in, values_in, keys_out, values_out, num_items, k, is_descending);
  REQUIRE(res == true);
}

C2H_TEST("DeviceTopK::TopKPairs: Test for large num_items", "[pairs][topk][device]", num_items_types)
{
  using key_t       = uint32_t;
  using value_t     = uint32_t;
  using num_items_t = c2h::get<0, TestType>;

  // Set input size
  constexpr num_items_t max_num_items_ull =
    cuda::std::min(static_cast<size_t>(cuda::std::numeric_limits<num_items_t>::max()),
                   cuda::std::numeric_limits<cuda::std::uint32_t>::max() + static_cast<size_t>(2000000ULL));
  constexpr num_items_t max_num_items = static_cast<num_items_t>(max_num_items_ull);
  const num_items_t num_items         = GENERATE_COPY(values({max_num_items}));

  // Set the k value
  constexpr num_items_t min_k = 1 << 3;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, min(num_items - 1, max_k))));

  // Prepare input and output
  auto keys_in   = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  auto values_in = cuda::make_counting_iterator(value_t{});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));
  c2h::device_vector<value_t> values_out(k, static_cast<value_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  bool res                 = topk_with_iterator(keys_in, values_in, keys_out, values_out, num_items, k, is_descending);
  REQUIRE(res == true);
}

C2H_TEST("DeviceTopK::TopKPairs: Test for different data types for num_items and k",
         "[pairs][topk][device]",
         k_items_types)
{
  using key_t       = uint32_t;
  using value_t     = uint32_t;
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
  auto keys_in   = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  auto values_in = cuda::make_counting_iterator(value_t{});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));
  c2h::device_vector<value_t> values_out(k, static_cast<value_t>(42));

  // Run the device-wide API
  const bool is_descending = GENERATE(false, true);
  bool res                 = topk_with_iterator(keys_in, values_in, keys_out, values_out, num_items, k, is_descending);
  REQUIRE(res == true);
}
