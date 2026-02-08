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
#include "catch2_radix_sort_helper.cuh"
#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include "cuda/__iterator/tabulate_output_iterator.h"
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
  auto stream_env = cuda::stream_ref{stream};
  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);

  auto env = cuda::std::execution::env{stream_env, requirements};

  auto values_it = static_cast<cub::NullType*>(nullptr);
  return cub::detail::dispatch_topk_hub<SelectDirection>(
    d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, values_it, values_it, num_items, k, env);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(dispatch_topk_keys, topk_keys, cub::detail::topk::select SelectDirection, SelectDirection);

using directions =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;
using num_items_types = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using k_items_types   = c2h::type_list<cuda::std::uint32_t, cuda::std::uint64_t>;
using key_types =
  c2h::type_list<cuda::std::uint8_t,
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

C2H_TEST("DeviceTopK::{Min,Max}Keys work as expected", "[keys][topk][device]", key_types, directions)
{
  using key_t              = c2h::get<0, TestType>;
  constexpr auto direction = c2h::get<1, TestType>::value;
  using num_items_t        = cuda::std::uint32_t;
  using comparator_t       = direction_to_comparator_t<direction>;

  // Set input size
  constexpr num_items_t min_num_items = 1;
  constexpr num_items_t max_num_items = 1 << 22;
  const num_items_t num_items         = GENERATE_COPY(
    values({min_num_items, num_items_t{3}, max_num_items}), take(1, random(min_num_items, max_num_items)));

  // Set the k value
  constexpr num_items_t min_k = 1;
  const num_items_t k         = GENERATE_COPY(take(3, random(min_k, num_items)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, direction);

  // Prepare input & output
  c2h::device_vector<key_t> keys_in(num_items);
  c2h::device_vector<key_t> keys_out(k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in);
  c2h::device_vector<key_t> expected_keys(keys_in);

  // Run the top-k algorithm
  topk_keys<direction>(
    thrust::raw_pointer_cast(keys_in.data()), thrust::raw_pointer_cast(keys_out.data()), num_items, k);

  // Sort the entire input data as result reference
  thrust::sort(expected_keys.begin(), expected_keys.end(), comparator_t{});
  expected_keys.resize(k);

  // Since the results of top-k are unordered, we need to sort the results before comparison.
  thrust::sort(keys_out.begin(), keys_out.end(), comparator_t{});

  REQUIRE(expected_keys == keys_out);
}

C2H_TEST("DeviceTopK::{Min,Max}Keys work with iterators", "[keys][topk][device]", key_types)
{
  using key_t              = c2h::get<0, TestType>;
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

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, direction);

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(num_items_t{}), inc_t<key_t>{static_cast<cuda::std::size_t>(num_items)});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the top-k algorithm
  topk_keys<direction>(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);

  // Verify results
  thrust::sort(keys_out.begin(), keys_out.end(), comparator_t{});
  if constexpr (direction == cub::detail::topk::select::max)
  {
    auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
    REQUIRE(thrust::equal(keys_out.cbegin(), keys_out.cend(), keys_expected_it));
  }
  else
  {
    REQUIRE(thrust::equal(keys_out.cbegin(), keys_out.cend(), keys_in));
  }
}

C2H_TEST("DeviceTopK::{Min,Max}Keys works with a large number of items", "[keys][topk][device]", num_items_types)
try
{
  using key_t              = cuda::std::uint32_t;
  using num_items_t        = c2h::get<0, TestType>;
  constexpr auto direction = cub::detail::topk::select::max;
  using comparator_t       = direction_to_comparator_t<direction>;

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

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), num_items, k, direction);

  // Prepare input and output
  auto keys_in = cuda::make_transform_iterator(
    cuda::make_counting_iterator(num_items_t{0}), inc_t<key_t>{static_cast<cuda::std::size_t>(num_items)});
  c2h::device_vector<key_t> keys_out(k, static_cast<key_t>(42));

  // Run the top-k algorithm
  topk_keys<direction>(keys_in, thrust::raw_pointer_cast(keys_out.data()), num_items, k);

  // Verify results
  thrust::sort(keys_out.begin(), keys_out.end(), comparator_t{});
  auto keys_expected_it = cuda::std::make_reverse_iterator(keys_in + num_items);
  REQUIRE(thrust::equal(keys_out.cbegin(), keys_out.cend(), keys_expected_it));
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}

C2H_TEST("DeviceTopK::{Min,Max}Keys works for different offset types for num_items and k",
         "[keys][topk][device]",
         k_items_types)
try
{
  using key_t              = cuda::std::uint32_t;
  using num_items_t        = cuda::std::uint64_t;
  using k_items_t          = c2h::get<0, TestType>;
  constexpr auto direction = cub::detail::topk::select::min;
  using comparator_t       = direction_to_comparator_t<direction>;

  // Set input size
  const num_items_t num_items = detail::make_large_offset<num_items_t>();

  // Set the k value
  const auto limit_k    = static_cast<k_items_t>(cuda::std::min(
    static_cast<num_items_t>(cuda::std::numeric_limits<k_items_t>::max()), static_cast<num_items_t>(num_items)));
  const k_items_t min_k = limit_k > k_items_t{10000} ? limit_k - k_items_t{10000} : k_items_t{1};
  const k_items_t k     = GENERATE_COPY(take(3, random(min_k, limit_k)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(), c2h::type_name<num_items_t>(), c2h::type_name<k_items_t>(), num_items, k, direction);

  // Prepare input
  auto counting_it = cuda::make_counting_iterator(num_items_t{});
  auto keys_in     = cuda::std::make_reverse_iterator(counting_it + num_items);

  // Prepare helper to check results
  auto check_result_helper = check_unordered_output_helper(k);
  auto check_result_it     = check_result_helper.get_flagging_output_iterator();

  // Run the top-k algorithm
  topk_keys<direction>(keys_in, check_result_it, num_items, k);

  // Verify results
  check_result_helper.check_all_results_correct();
}
catch (std::bad_alloc& e)
{
  std::cerr << "Caught bad_alloc: " << e.what() << std::endl;
}
