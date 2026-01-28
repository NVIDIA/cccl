// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/sort.h>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include "catch2_test_device_topk_common.cuh"
#include "catch2_test_launch_helper.h"
#include "cuda/std/__algorithm/min.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <catch2/generators/catch_generators.hpp>

template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename SelectDirectionParamT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  SelectDirectionParamT select_directions,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream = 0)
{
  using value_it_t = cub::NullType**;

  auto values_it = static_cast<cub::NullType**>(nullptr);
  return cub::detail::batched_topk::dispatch_batched_topk<
    KeyInputItItT,
    KeyOutputItItT,
    value_it_t,
    value_it_t,
    SegmentSizeParamT,
    KParamT,
    SelectDirectionParamT,
    NumSegmentsParameterT,
    TotalNumItemsGuaranteeT>::
    dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      values_it,
      values_it,
      segment_sizes,
      k,
      select_directions,
      num_segments,
      total_num_items_guarantee,
      stream);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(dispatch_batched_topk_keys, batched_topk_keys);

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


C2H_TEST("DeviceSegmentedTopK::{Min,Max}Keys work with small fixed-size segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_size_list,
         max_num_k_list)
{
  using segment_size_t  = cuda::std::int64_t;
  using segment_index_t = cuda::std::int64_t;

  using key_t = c2h::get<0, TestType>;

  // Statically constrained maximum segment size and k
  constexpr segment_size_t static_max_segment_size = c2h::get<1, TestType>::value;
  constexpr segment_size_t static_max_k            = c2h::get<2, TestType>::value;

  // Test both directions (as runtime value)
  const auto direction = GENERATE_COPY(cub::detail::topk::select::min, cub::detail::topk::select::max);

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size           = static_max_segment_size;
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(4, random(min_segment_size, max_segment_size)));
  const segment_size_t max_k        = cuda::std::min(static_max_k, segment_size);

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({segment_size_t{1}, max_k}), take(3, random(segment_size_t{1}, max_k)));

  // Generate number of segments
  const segment_index_t num_segments = GENERATE_COPY(
    values({segment_index_t{1}, segment_index_t{42}}), take(4, random(segment_index_t{1}, segment_index_t{10000})));

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
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size);
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
  batched_topk_keys(
    d_keys_in,
    d_keys_out,
    cub::detail::batched_topk::segment_size_uniform<1, max_segment_size>{segment_size},
    cub::detail::batched_topk::k_uniform<1, static_max_k>{k},
    cub::detail::batched_topk::select_direction_uniform{direction},
    cub::detail::batched_topk::num_segments_uniform<>{num_segments},
    cub::detail::batched_topk::total_num_items_guarantee{num_segments * segment_size});
  // Prepare expected results
  segmented_sort_keys(expected_keys, num_segments, segment_size, direction);
  compact_sorted_keys_to_topk(expected_keys, segment_size, k);

  // Since the results of top-k are unordered, sort output segments before comparison.
  segmented_sort_keys(keys_out_buffer, num_segments, k, direction);

  REQUIRE(expected_keys == keys_out_buffer);
}
