// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_topk.cuh>

#include <thrust/memory.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>

#include <algorithm>

#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

template <cub::detail::topk::select SelectDirection,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename NumItemsT,
          typename NumOutItemsT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputIteratorT d_keys_in,
  KeyOutputIteratorT d_keys_out,
  NumItemsT num_items,
  NumOutItemsT k,
  cudaStream_t stream = 0)
{
  auto stream_env = cuda::std::execution::prop{cuda::get_stream_t{}, cuda::stream_ref{stream}};
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  auto env = cuda::std::execution::env{stream_env, requirements};

  auto values_it = static_cast<cub::NullType*>(nullptr);
  return cub::detail::dispatch_topk_hub<SelectDirection>(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, values_it, values_it, num_items, k, env);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(dispatch_topk_keys<cub::detail::topk::select::max>, topk_max_keys);
DECLARE_LAUNCH_WRAPPER(dispatch_topk_keys<cub::detail::topk::select::min>, topk_min_keys);

using key_types =
  c2h::type_list<cuda::std::uint8_t,
                 cuda::std::uint16_t,
                 float,
                 cuda::std::uint64_t
// clang-format off
#if TEST_HALF_T()
                 , half_t
#endif // TEST_HALF_T()
#if TEST_BF_T()
                 , bfloat16_t
#endif // TEST_BF_T()
>;
// clang-format on
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using k_items_types   = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
C2H_TEST("DeviceTopK::MaxKeys: Basic testing", "[keys][topk][device]", key_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = cuda::std::uint32_t;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 15;
  const num_items_t num_items =
    GENERATE(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, cuda::std::min(num_items, max_k))));

  // Whether to select k elements with the lowest or the highest values
  const bool is_descending = GENERATE(false, true);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, is_descending);

  // Prepare input & output
  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(k);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);
  c2h::host_vector<key_t> h_keys_in(keys_in);

  // Run the device-wide API
  if (!is_descending)
  {
    topk_min_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_max_keys(thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  c2h::host_vector<key_t> expected_results(keys_out.size());
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    // Sort the entire input data as result reference
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), expected_results.begin(), expected_results.end(), std::greater<key_t>());

    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
  }
  else
  {
    // Sort the entire input data as result reference
    std::partial_sort_copy(
      h_keys_in.begin(), h_keys_in.end(), expected_results.begin(), expected_results.end(), std::less<key_t>());

    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
  }

  REQUIRE(expected_results == device_results);
}

C2H_TEST("DeviceTopK::MaxKeys: works with iterators", "[keys][topk][device]", key_types)
{
  using key_t       = c2h::get<0, TestType>;
  using num_items_t = cuda::std::uint32_t;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 24;
  const num_items_t num_items =
    GENERATE(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, cuda::std::min(num_items, max_k))));

  // Whether to select k elements with the lowest or the highest values
  const bool is_descending = GENERATE(false, true);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, is_descending);

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{static_cast<cuda::std::size_t>(num_items)});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-wide API
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_max_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_expected_it));
  }
  else
  {
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_in));
  }
}

C2H_TEST("DeviceTopK::MaxKeys: Test for large num_items", "[keys][topk][device]", num_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using num_items_t = c2h::get<0, TestType>;

  // Set input size
  constexpr auto max_num_items_ull =
    cuda::std::min(static_cast<cuda::std::size_t>(cuda::std::numeric_limits<num_items_t>::max()),
                   cuda::std::numeric_limits<cuda::std::uint32_t>::max() + static_cast<cuda::std::size_t>(2000000ULL));
  constexpr num_items_t max_num_items = static_cast<num_items_t>(max_num_items_ull);
  const num_items_t num_items         = GENERATE_COPY(values({max_num_items}));

  // Set the k value
  constexpr num_items_t min_k = 1;
  constexpr num_items_t max_k = 1 << 15;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, cuda::std::min(num_items, max_k))));

  // Whether to select k elements with the lowest or the highest values
  const bool is_descending = GENERATE(false, true);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, is_descending);

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{static_cast<cuda::std::size_t>(num_items)});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-side top-k
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_max_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());

    // Skip materialization of verification data for performance reasons in large-num_items case
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_expected_it));
  }
  else
  {
    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());

    // Skip materialization of verification data for performance reasons in large-num_items case
    REQUIRE(thrust::equal(device_results.cbegin(), device_results.cend(), keys_in));
  }
}

C2H_TEST("DeviceTopK::MaxKeys:  Test for different data types for num_items and k",
         "[keys][topk][device]",
         k_items_types)
{
  using key_t       = cuda::std::uint32_t;
  using num_items_t = cuda::std::uint32_t;
  using k_items_t   = c2h::get<0, TestType>;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 20;
  const num_items_t num_items =
    GENERATE(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr k_items_t min_k = 1;
  k_items_t limit_k = cuda::std::min(cuda::std::numeric_limits<k_items_t>::max(), static_cast<k_items_t>(1 << 15));
  const k_items_t max_k =
    num_items < cuda::std::numeric_limits<k_items_t>::max()
      ? cuda::std::min(static_cast<k_items_t>(num_items), limit_k)
      : limit_k;
  const k_items_t k = GENERATE_COPY(take(3, random(min_k, max_k)));

  // Whether to select k elements with the lowest or the highest values
  const bool is_descending = GENERATE(false, true);

  // Capture test parameters
  CAPTURE(
    c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), c2h::type_name<k_items_t>(), num_items, k, is_descending);

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{static_cast<cuda::std::size_t>(num_items)});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the device-side top-k
  if (!is_descending)
  {
    topk_min_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }
  else
  {
    topk_max_keys(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);
  }

  // Verify results
  c2h::host_vector<key_t> device_results(keys_out);
  if (is_descending)
  {
    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::greater<key_t>());
    auto keys_reversed_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    c2h::device_vector<key_t> expected_keys(keys_reversed_it, keys_reversed_it + k);
    REQUIRE(device_results == expected_keys);
  }
  else
  {
    // Since the results of top-k are unordered, we need to sort the results before comparison.
    std::stable_sort(device_results.begin(), device_results.end(), std::less<key_t>());
    c2h::device_vector<key_t> expected_keys(keys_in, keys_in + k);
    REQUIRE(device_results == expected_keys);
  }
}
