// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/util_type.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/scan.h>

#include <cuda/iterator>
#include <cuda/std/__algorithm/min.h>

#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <catch2/generators/catch_generators.hpp>

struct is_minus_zero
{
  __device__ bool operator()(float x) const
  {
    return x == 0.0f && cuda::std::signbit(x);
  }
};

template <cub::detail::topk::select Direction,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream = nullptr)
{
  auto values_it = static_cast<cub::NullType**>(nullptr);
  return cub::detail::batched_topk::dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_key_segments_it,
    d_key_segments_out_it,
    values_it,
    values_it,
    segment_sizes,
    k,
    cuda::args::constant<Direction>{},
    num_segments,
    total_num_items_guarantee,
    stream);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_batched_topk_keys, batched_topk_keys, cub::detail::topk::select Direction, Direction);

// Total segment size
using max_segment_size_list = c2h::enum_type_list<cuda::std::size_t, 4 * 1024>;

// Segment size: static, uniform
using max_num_k_list = c2h::enum_type_list<cuda::std::size_t, 32, 4 * 1024>;

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

// Selection direction is a compile-time option; cover both as a static test axis.
using select_direction_list =
  c2h::enum_type_list<cub::detail::topk::select, cub::detail::topk::select::min, cub::detail::topk::select::max>;

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with small fixed-size segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(4, random(min_segment_size, max_segment_size)));
  const segment_size_t max_k        = (cuda::std::min) (static_max_k, segment_size);

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, max_k}), take(3, random(segment_size_t{1}, max_k)));

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(4, random(segment_index_t{1}, segment_index_t{1000})));

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

  // Prepare input & output
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    cuda::args::immediate{num_segments * segment_size});
  // Prepare expected results
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  fixed_size_segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with small variable-size segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

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
    get_output_size_op{segment_offsets.cbegin(), cuda::constant_iterator(k), num_segments});
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

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  batched_topk_keys<direction>(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::immediate{num_segments},
    cuda::args::immediate{num_items});

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, k);

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with fixed-size segments and per-segment k",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

  // Generate the (uniform) input segment size. Unlike the uniform-k tests, k still varies per segment below.
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(2, random(min_segment_size, max_segment_size)));

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(2, random(segment_index_t{1}, segment_index_t{1000})));

  // Generate a per-segment k in [1, static_max_k]
  c2h::device_vector<segment_size_t> segment_k(num_segments, thrust::no_init);
  c2h::gen(C2H_SEED(1), segment_k, segment_size_t{1}, static_max_k);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          segment_size,
          num_segments,
          direction);

  // Materialize fixed-size input offsets: [0, segment_size, 2 * segment_size, ...]
  auto fixed_offsets_it = cuda::make_strided_iterator(cuda::make_counting_iterator<segment_size_t>(0), segment_size);
  c2h::device_vector<segment_size_t> segment_offsets(num_segments + 1, thrust::no_init);
  thrust::copy(fixed_offsets_it, fixed_offsets_it + (num_segments + 1), segment_offsets.begin());

  // Compute compacted output offsets: each output segment holds exactly min(k[i], segment_size) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), segment_k.cbegin(), num_segments});
  c2h::device_vector<segment_size_t> compacted_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(
    compacted_output_sizes_it, compacted_output_sizes_it + num_segments + 1, compacted_offsets.begin());
  segment_size_t total_output_size = compacted_offsets.back();

  // Prepare input & output. Input segments are fixed-size (strided); output segments are compacted (variable).
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size, thrust::no_init);
  c2h::device_vector<key_t> keys_out_buffer(total_output_size, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out =
    cuda::make_permutation_iterator(cuda::make_counting_iterator(d_keys_out_ptr), compacted_offsets.cbegin());

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as an immediate sequence
  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    cuda::args::immediate{segment_size, cuda::args::bounds<segment_size_t{1}, max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::constant<direction>{},
    cuda::args::immediate{num_segments},
    cuda::args::immediate{num_segments * segment_size});

  // Prepare expected results: sort each fixed-size input segment, then compact each to its per-segment top-k.
  fixed_size_segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison.
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

C2H_TEST("DeviceBatchedTopK::{Min,Max}Keys work with variable-size segments and per-segment k",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list,
         select_direction_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Selection direction comes from the compile-time test axis.
  constexpr auto direction = c2h::get<3, TestType>::value;

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

  // Generate a per-segment k in [1, static_max_k]
  c2h::device_vector<segment_size_t> segment_k(num_segments, thrust::no_init);
  c2h::gen(C2H_SEED(1), segment_k, segment_size_t{1}, static_max_k);

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_index_t>(),
          static_max_segment_size,
          static_max_k,
          num_segments,
          direction);

  // Compute compacted output offsets:
  // Each output segment holds exactly min(k[i], segment_size[i]) items, tightly packed.
  auto compacted_output_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(segment_index_t{0}),
    get_output_size_op{segment_offsets.cbegin(), segment_k.cbegin(), num_segments});
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

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm with a per-segment k passed as an immediate sequence
  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    cuda::args::deferred_sequence{segment_size_it, cuda::args::bounds<segment_size_t{1}, static_max_segment_size>()},
    cuda::args::deferred_sequence{
      thrust::raw_pointer_cast(segment_k.data()), cuda::args::bounds<segment_size_t{1}, static_max_k>()},
    cuda::args::constant<direction>{},
    cuda::args::immediate{num_segments},
    cuda::args::immediate{num_items});

  // Verify keys are returned correctly: sort each segment of the expected input, then compact the per-segment top-k
  segmented_sort_keys(expected_keys, num_segments, segment_offsets.cbegin(), segment_offsets.cbegin() + 1, direction);
  expected_keys = compact_to_topk_batched(expected_keys, segment_offsets, segment_k.cbegin());

  // Since the results of top-k are unordered, sort compacted output segments before comparison
  segmented_sort_keys(
    keys_out_buffer, num_segments, compacted_offsets.cbegin(), compacted_offsets.cbegin() + 1, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}

// Regression test: top-k must preserve -0.0f in the output (not normalize to +0.0f).
C2H_TEST("DeviceBatchedTopK::MinKeys preserves -0.0f in output", "[keys][segmented][topk][device][float]")
{
  constexpr cuda::std::int64_t segment_size                      = 8;
  constexpr cuda::std::int64_t k                                 = 5;
  constexpr cuda::std::int64_t num_segments                      = 1;
  [[maybe_unused]] constexpr cuda::std::int64_t max_segment_size = 64; // msvc warns, only used in nttp

  // Input: one segment containing -0.0f and +0.0f; top-5 min should include both zeros.
  c2h::device_vector<float> d_keys_in{3.0f, -0.0f, 1.0f, 2.0f, 0.0f, -1.0f, 4.0f, 5.0f};
  c2h::device_vector<float> d_keys_out(k, thrust::no_init);

  auto d_keys_in_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_in.data())), segment_size);
  auto d_keys_out_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator(thrust::raw_pointer_cast(d_keys_out.data())), k);

  batched_topk_keys<cub::detail::topk::select::min>(
    d_keys_in_it,
    d_keys_out_it,
    cuda::args::immediate{segment_size, cuda::args::bounds<cuda::std::int64_t{1}, max_segment_size>()},
    cuda::args::immediate{k, cuda::args::bounds<cuda::std::int64_t{1}, k>()},
    cuda::args::immediate{num_segments},
    cuda::args::immediate{num_segments * segment_size});

  const int num_minus_zero = static_cast<int>(thrust::count_if(d_keys_out.begin(), d_keys_out.end(), is_minus_zero{}));
  REQUIRE(num_minus_zero >= 1);
}
