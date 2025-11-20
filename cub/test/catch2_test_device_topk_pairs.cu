// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_topk.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>
#include <cuda/std/type_traits>

#include "catch2_large_problem_helper.cuh"
#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

template <cub::detail::topk::select SelectDirection,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          typename NumOutItemsT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_topk_pairs(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputIteratorT d_keys_in,
  KeyOutputIteratorT d_keys_out,
  ValueInputIteratorT d_values_in,
  ValueOutputIteratorT d_values_out,
  NumItemsT num_items,
  NumOutItemsT k,
  cudaStream_t stream = 0)
{
  auto stream_env = cuda::stream_ref{stream};
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  auto env = cuda::std::execution::env{stream_env, requirements};

  return cub::detail::dispatch_topk_hub<SelectDirection>(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, env);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(dispatch_topk_pairs, topk_pairs, cub::detail::topk::select SelectDirection, SelectDirection);

template <typename KeyT, typename ValueT, typename NumItemsT, typename ComperatorT>
void sort_keys_and_values(
  c2h::device_vector<KeyT>& keys, c2h::device_vector<ValueT>& values, NumItemsT num_items, ComperatorT comperator_op)
{
  // Perform sort: primary sort on the keys, secondary sort on the values
  thrust::sort_by_key(values.begin(), values.begin() + num_items, keys.begin(), comperator_op);
  thrust::sort_by_key(keys.begin(), keys.begin() + num_items, values.begin(), comperator_op);
}

template <typename KeyInItT, typename ValueInItT, typename KeyT, typename ValueT, typename NumItemsT, typename ComperatorT>
bool check_results(
  KeyInItT h_keys_in,
  ValueInItT h_values_in,
  c2h::device_vector<KeyT>& keys_out,
  c2h::device_vector<ValueT>& values_out,
  NumItemsT num_items,
  NumItemsT k,
  ComperatorT comperator_op)
{
  // Since the results of API MinPairs() and MaxPairs() are not-sorted, we need to sort the results first.
  sort_keys_and_values(keys_out, values_out, k, comperator_op);
  c2h::host_vector<KeyT> h_keys_out(keys_out);
  c2h::host_vector<ValueT> h_values_out(values_out);

  // i for results from gpu (MinPairs() and MaxPairs()); j for reference results
  NumItemsT i = 0, j = 0;
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
      // if (is_descending ? h_values_out[i] < h_values_in[j] : h_values_out[i] > h_values_in[j])
      else if (comperator_op(h_values_in[j], h_values_out[i]))
      {
        // Note: The results returned by the API functions MinPairs() and MaxPairs() are not stable.
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

using directions =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;
using key_types       = c2h::type_list<cuda::std::uint16_t, float, cuda::std::uint64_t>;
using value_types     = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using k_items_types   = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;

C2H_TEST("DeviceTopK::MaxPairs: Basic testing", "[pairs][topk][device]", key_types, value_types, directions)
{
  using key_t              = c2h::get<0, TestType>;
  using value_t            = c2h::get<1, TestType>;
  constexpr auto direction = c2h::get<2, TestType>::value;
  using num_items_t        = cuda::std::uint32_t;
  using comperator_t       = direction_to_comparator_t<direction>;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 22;
  const num_items_t num_items =
    GENERATE_COPY(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, num_items)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<value_t>(), c2h::type_name<num_items_t>(), num_items, k, direction);

  // Allocate device memory
  c2h::device_vector<key_t> keys_in(num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out(k, thrust::no_init);
  c2h::device_vector<value_t> values_in(num_items, thrust::no_init);
  c2h::device_vector<value_t> values_out(k, thrust::no_init);

  // Initialize input data
  const int num_key_seeds   = 1;
  const int num_value_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);
  c2h::gen(C2H_SEED(num_value_seeds), values_in);

  topk_pairs<direction>(
    thrust::raw_pointer_cast(keys_in.data()),
    thrust::raw_pointer_cast(keys_out.data()),
    thrust::raw_pointer_cast(values_in.data()),
    thrust::raw_pointer_cast(values_out.data()),
    num_items,
    k);

  // Sort the entire input data as result reference
  sort_keys_and_values(keys_in, values_in, num_items, comperator_t{});
  c2h::host_vector<key_t> h_keys(keys_in);
  c2h::host_vector<value_t> h_values(values_in);

  const bool res = check_results(
    thrust::raw_pointer_cast(h_keys.data()),
    thrust::raw_pointer_cast(h_values.data()),
    keys_out,
    values_out,
    num_items,
    k,
    comperator_t{});

  REQUIRE(res == true);
}

C2H_TEST("DeviceTopK::MaxPairs: Works with iterators", "[pairs][topk][device]", key_types, value_types)
{
  using key_t              = c2h::get<0, TestType>;
  using value_t            = c2h::get<1, TestType>;
  using num_items_t        = cuda::std::uint32_t;
  constexpr auto direction = cub::detail::topk::select::max;
  using comparator_t       = direction_to_comparator_t<direction>;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 24;
  const num_items_t num_items =
    GENERATE_COPY(values({min_num_items, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, num_items)));

  // Prepare input and output
  auto keys_in   = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  auto values_in = cuda::make_counting_iterator(value_t{});
  c2h::device_vector<key_t> keys_out(k, key_t{42});
  c2h::device_vector<value_t> values_out(k, value_t{42});

  // Run the top-k algorithm
  topk_pairs<direction>(
    keys_in,
    thrust::raw_pointer_cast(keys_out.data()),
    values_in,
    thrust::raw_pointer_cast(values_out.data()),
    num_items,
    k);

  // Verify results
  const auto keys_expected_it   = cuda::std::make_reverse_iterator(keys_in + num_items);
  const auto values_expected_it = cuda::std::make_reverse_iterator(values_in + num_items);
  const bool res                = check_results(
    keys_expected_it, values_expected_it, keys_out, values_out, num_items, static_cast<num_items_t>(k), comparator_t{});
  REQUIRE(res == true);
}

C2H_TEST("DeviceTopK::MaxPairs: Test for large num_items", "[pairs][topk][device]", num_items_types)
{
  using key_t              = cuda::std::uint32_t;
  using value_t            = cuda::std::uint32_t;
  using num_items_t        = c2h::get<0, TestType>;
  constexpr auto direction = cub::detail::topk::select::max;
  using comparator_t       = direction_to_comparator_t<direction>;

  // Set input size
  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const num_items_t num_items_max = detail::make_large_offset<num_items_t>();
  const num_items_t num_items_min = num_items_max > 10000 ? num_items_max - 10000ULL : num_items_t{0};
  const num_items_t num_items =
    GENERATE_COPY(values({num_items_max, static_cast<num_items_t>(num_items_max - 1), num_items_t{1}, num_items_t{3}}),
                  take(2, random(num_items_min, num_items_max)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  constexpr num_items_t max_k = 1 << 20;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, cuda::std::min(num_items, max_k))));

  // Prepare input and output
  auto keys_in   = cuda::make_transform_iterator(cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{num_items});
  auto values_in = cuda::make_counting_iterator(value_t{});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));
  c2h::device_vector<value_t> values_out(k, static_cast<value_t>(42));

  // Run the top-k algorithm
  topk_pairs<direction>(
    keys_in,
    thrust::raw_pointer_cast(keys_out.data()),
    values_in,
    thrust::raw_pointer_cast(values_out.data()),
    num_items,
    k);

  // Verify results
  auto keys_expected_it   = cuda::std::make_reverse_iterator(keys_in + num_items);
  auto values_expected_it = cuda::std::make_reverse_iterator(values_in + num_items);
  bool res                = check_results(
    keys_expected_it, values_expected_it, keys_out, values_out, num_items, static_cast<num_items_t>(k), comparator_t{});
  REQUIRE(res == true);
}
