// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <cuda/iterator>

#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <catch2/generators/catch_generators.hpp>

// Maps an item index to its segment id for fixed-size segments
struct fixed_stride_segment_id_op
{
  cuda::std::int64_t stride;

  template <typename IndexT>
  __device__ IndexT operator()(IndexT idx) const
  {
    return static_cast<IndexT>(idx / stride);
  }
};

// Flags adjacent duplicate items that belong to the same segment
template <typename ItemItT, typename SegIdItT>
struct flag_intra_segment_duplicates
{
  ItemItT d_sorted_items;
  SegIdItT d_segment_ids;

  template <typename IndexT>
  __device__ bool operator()(IndexT idx) const
  {
    return d_segment_ids[idx] == d_segment_ids[idx + 1] && d_sorted_items[idx] == d_sorted_items[idx + 1];
  }
};

template <typename ItemItT, typename SegIdItT>
flag_intra_segment_duplicates(ItemItT, SegIdItT) -> flag_intra_segment_duplicates<ItemItT, SegIdItT>;

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::detail::batched_topk::dispatch, batched_topk_pairs);

// Total segment size
using max_segment_size_list = c2h::enum_type_list<cuda::std::size_t, 4 * 1024>;

// Segment size: static, uniform
using max_num_k_list = c2h::enum_type_list<cuda::std::size_t, 32, 4 * 1024>;

// %PARAM% TEST_TYPES types 0:1:2

#if TEST_TYPES == 0
using key_types =
  c2h::type_list<cuda::std::uint8_t
// clang-format off
  #if TEST_HALF_T()
  , half_t
  #endif // TEST_HALF_T()
  #if TEST_BF_T()
  , bfloat16_t
  #endif // TEST_BF_T()
  >;
// clang-format on
#elif TEST_TYPES == 1
using key_types = c2h::type_list<float>;
#elif TEST_TYPES == 2
using key_types = c2h::type_list<cuda::std::uint64_t>;
#endif

// Unsigned integer types used for the radix-pass boundary distribution test
using uint_key_types = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, cuda::std::uint64_t>;

// Consistency check: ensures values remain associated with their corresponding keys
template <typename KeyT, typename ValueT>
bool verify_pairs_consistency(const c2h::device_vector<KeyT>& keys_in,
                              const c2h::device_vector<KeyT>& keys_out,
                              const c2h::device_vector<ValueT>& values_out)
{
  auto d_keys_in    = thrust::raw_pointer_cast(keys_in.data());
  auto d_values_out = thrust::raw_pointer_cast(values_out.data());

  // permutation_it[i] -> d_keys_in[d_values_out[i]] to verify that keys and values remained associated
  auto permutation_it = cuda::make_permutation_iterator(d_keys_in, d_values_out);

  return thrust::equal(keys_out.cbegin(), keys_out.cend(), permutation_it);
}

// Uniqueness check: ensures there are no duplicate values within the top-k items of each segment
template <typename ValueT>
bool verify_unique_indices(c2h::device_vector<ValueT>& values_out, cuda::std::int64_t num_segments, cuda::std::int64_t k)
{
  // Make a copy & sort
  c2h::device_vector<ValueT> sorted_values{values_out};
  fixed_size_segmented_sort_keys(sorted_values, num_segments, k, cub::detail::topk::select::min);

  auto num_items   = sorted_values.size();
  auto counting_it = cuda::make_counting_iterator(cuda::std::int64_t{0});
  auto seg_ids     = cuda::make_transform_iterator(counting_it, fixed_stride_segment_id_op{k});
  flag_intra_segment_duplicates flag_op{sorted_values.cbegin(), seg_ids};
  auto num_duplicates = thrust::count_if(counting_it, counting_it + (num_items - 1), flag_op);

  return num_duplicates == 0;
}

// Overload for variable-size segments: sorts compacted values within each segment and checks for duplicates
template <typename ValueT, typename OffsetT>
bool verify_unique_indices(const c2h::device_vector<ValueT>& values_compacted,
                           const c2h::device_vector<OffsetT>& compacted_offsets,
                           cuda::std::int64_t num_segments)
{
  c2h::device_vector<ValueT> sorted_values = values_compacted;
  segmented_sort_keys(
    sorted_values,
    num_segments,
    compacted_offsets.cbegin(),
    compacted_offsets.cbegin() + 1,
    cub::detail::topk::select::min);

  auto num_items = sorted_values.size();

  // Generate segment ids via scatter + inclusive_scan: scatter a 1 at each interior segment
  // boundary, then prefix-sum to produce monotonic group ids
  c2h::device_vector<OffsetT> segment_ids(num_items, OffsetT{0});
  thrust::scatter(cuda::constant_iterator<OffsetT>(1),
                  cuda::constant_iterator<OffsetT>(1) + (num_segments - 1),
                  compacted_offsets.cbegin() + 1,
                  segment_ids.begin());
  thrust::inclusive_scan(segment_ids.begin(), segment_ids.end(), segment_ids.begin());

  flag_intra_segment_duplicates flag_op{sorted_values.cbegin(), segment_ids.cbegin()};

  auto num_duplicates =
    thrust::count_if(cuda::make_counting_iterator(size_t{0}), cuda::make_counting_iterator(num_items - 1), flag_op);

  return num_duplicates == 0;
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with small fixed-size segments",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Test both directions (as runtime value)
  const auto direction = GENERATE_COPY(cub::detail::topk::select::min, cub::detail::topk::select::max);

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(1, random(min_segment_size, max_segment_size)));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, max_k}), take(1, random(segment_size_t{1}, max_k)));

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(1, random(segment_index_t{1}, segment_index_t{1000})));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          segment_size,
          k,
          num_segments,
          direction);

  // Prepare keys input & output
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(num_segments * k, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(values_in_it), segment_size);
  auto d_values_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_values_out_ptr), k);

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_pairs(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cub::detail::batched_topk::segment_size_uniform<1, max_segment_size>{segment_size},
    cub::detail::batched_topk::k_uniform<1, static_max_k>{k},
    cub::detail::batched_topk::select_direction_uniform{direction},
    cub::detail::batched_topk::num_segments_uniform<>{num_segments},
    cub::detail::batched_topk::total_num_items_guarantee{num_segments * segment_size});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  // making sure we do not duplicate values Verify values remain associated with their corresponding keys
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  // This catches the case where we just returned a valid value multiple times
  REQUIRE(verify_unique_indices(values_out_buffer, num_segments, k) == true);

  // Verify keys are returned correctly
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Pairs work with small variable-size segments",
         "[pairs][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;
  using val_t = cuda::std::int32_t;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Test both directions (as runtime value)
  const auto direction = GENERATE_COPY(cub::detail::topk::select::min, cub::detail::topk::select::max);

  constexpr segment_size_t min_items = 1;
  constexpr segment_size_t max_items = 1'000'000;

  // Number of items
  const segment_size_t num_items = GENERATE_COPY(
    take(2, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  // Generate segment sizes
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  c2h::device_vector<segment_size_t> segment_offsets =
    c2h::gen_uniform_offsets<segment_size_t>(C2H_SEED(3), num_items, min_segment_size, max_segment_size);
  const segment_index_t num_segments = static_cast<segment_index_t>(segment_offsets.size() - 1);
  auto segment_offsets_it            = thrust::raw_pointer_cast(segment_offsets.data());
  auto segment_size_it               = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}), segment_size_op<segment_size_t*>{segment_offsets_it});

  // Set the k value
  const segment_size_t k =
    GENERATE_COPY(values({segment_size_t{1}, static_max_k}), take(3, random(segment_size_t{1}, static_max_k)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          k,
          num_segments,
          direction);

  // Compute compacted output offsets:
  // Each output segment holds exactly min(k, segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k)});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  // Prepare keys input & output
  c2h::device_vector<key_t> keys_in_buffer(num_items, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_offsets.cbegin());
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Prepare values input & output
  auto values_in_it = cuda::make_counting_iterator(val_t{0});
  c2h::device_vector<val_t> values_out_buffer(total_output_size, thrust::no_init);
  auto d_values_out_ptr = thrust::raw_pointer_cast(values_out_buffer.data());
  auto d_values_in =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(values_in_it), segment_offsets.cbegin());
  auto d_values_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_values_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_pairs(
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    cub::detail::batched_topk::segment_size_per_segment<decltype(segment_size_it), 1, static_max_segment_size>{
      segment_size_it},
    cub::detail::batched_topk::k_uniform<1, static_max_k>{k},
    cub::detail::batched_topk::select_direction_uniform{direction},
    cub::detail::batched_topk::num_segments_uniform<>{num_segments},
    cub::detail::batched_topk::total_num_items_guarantee{num_items});

  // Verification:
  // - We verify correct top-k selection through the keys
  // - We verify that values were permuted along correctly by making sure values remain associated with their keys and
  //   making sure we do not duplicate values
  REQUIRE(verify_pairs_consistency(expected_keys, keys_out_buffer, values_out_buffer) == true);

  // Verify values don't appear more than once in the returned results
  REQUIRE(verify_unique_indices(values_out_buffer, compacted_offsets, num_segments) == true);

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
