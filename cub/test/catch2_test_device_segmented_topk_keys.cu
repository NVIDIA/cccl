// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/dispatch/dispatch_segmented_topk.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
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
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_segmented_topk_keys(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  SegmentSizeParamT segment_sizes,
  KParamT k_it,
  SelectDirectionParamT select_directions,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream = 0)
{
  using value_it_t = cub::NullType**;

  auto values_it = static_cast<cub::NullType**>(nullptr);
  return cub::detail::topk::DispatchSegmentedTopK<
    KeyInputItItT,
    KeyOutputItItT,
    value_it_t,
    value_it_t,
    SegmentSizeParamT,
    KParamT,
    SelectDirectionParamT,
    NumSegmentsParameterT,
    TotalNumItemsGuaranteeT>::
    Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k_it,
      select_directions,
      num_segments,
      total_num_items_guarantee,
      stream);
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_TMPL_LAUNCH_WRAPPER(
  dispatch_segmented_topk_keys, segmented_topk_keys, cub::detail::topk::select SelectDirection, SelectDirection);

// Total segment size in bytes
using max_segment_bytes_list_t = c2h::enum_type_list<cuda::std::size_t, 2048, 45056>;

// Segment size: static, uniform
using max_num_k_list = c2h::enum_type_list<cuda::std::size_t, 1, 8, 32, 128, 45056>;

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

// namespace Catch
// {
// template <typename T>
// struct StringMaker<cub::detail::transform::aligned_base_ptr<T>>
// {
//   static auto convert(cub::detail::transform::aligned_base_ptr<T> abp) -> std::string
//   {
//     std::stringstream ss;
//     ss << "{ptr: " << abp.ptr << ", head_padding: " << abp.head_padding << "}";
//     return ss.str();
//   }
// };
// } // namespace Catch

template <typename KeyT, typename ComparatorT>
void segmented_sort_keys(
  c2h::device_vector<KeyT>& d_keys_in,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t segment_size,
  cuda::std::int64_t k,
  ComparatorT comparator)
{
  cuda::std::int64_t num_items = d_keys_in.size();
  // Prepare alternate buffer for double buffering
  c2h::device_vector<KeyT> d_keys_alt(num_items, thrust::no_init);
  cub::DoubleBuffer<KeyT> d_keys(
    thrust::raw_pointer_cast(d_keys_in.data()), thrust::raw_pointer_cast(d_keys_alt.data()));

  // Prepare segment offsets
  auto segment_offsets_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator<cuda::std::int64_t>(0), segment_size);

  // Query temporary storage size
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedSort::SortKeys(
    nullptr, temp_storage_bytes, d_keys, num_items, num_segments, segment_offsets_it, (segment_offsets_it + 1));

  // Allocate temporary storage
  c2h::device_vector<::cuda::std::uint8_t> d_temp_storage(temp_storage_bytes);

  // Run segmented sort
  cub::DeviceSegmentedSort::SortKeys(
    thrust::raw_pointer_cast(d_temp_storage.data()),
    temp_storage_bytes,
    d_keys,
    num_items,
    num_segments,
    segment_offsets_it,
    (segment_offsets_it + 1));

  // Make sure the result is returned in the original buffer
  if (d_keys.Current() != thrust::raw_pointer_cast(d_keys_in.data()))
  {
    thrust::copy(d_keys.Current(), d_keys.Current() + num_items, d_keys_in.begin());
  }
}

// make this function object:
struct remove_out_of_topk_op
{
  cuda::std::int64_t segment_size;
  cuda::std::int64_t k;

  bool __device__ operator()(cuda::std::int64_t idx) const
  {
    auto offset_in_segment = idx % segment_size;
    return offset_in_segment >= k;
  }
};

template <typename ComparatorT>
void compact_sorted_keys_to_topk(
  c2h::device_vector<key_t>& d_keys_in,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t segment_size,
  cuda::std::int64_t k)
{
  auto new_end = thrust::remove_if(
    d_keys_in.begin(),
    d_keys_in.end(),
    cuda::make_counting_iterator(0),
    remove_out_of_topk_op{segment_size, k});

  // Resize input to new size
  d_keys_in.resize(new_end - d_keys_in.begin());
}

C2H_TEST("DeviceSegmentedTopK::{Min,Max}Keys work with small fixed-sized segments",
         "[keys][segmented][topk][device]",
         key_types,
         max_segment_bytes_list_t,
         max_num_k_list)
{
  using segment_size_t = cuda::std::int32_t;
  using segment_index_t = cuda::std::int32_t;

  using key_t                      = c2h::get<0, TestType>;
  constexpr auto max_segment_bytes = c2h::get<1, TestType>::value;
  constexpr segment_size_t max_num_k         = c2h::get<2, TestType>::value;

  // Test both directions (as runtime value)
  const auto direction = GENERATE_COPY(cub::detail::topk::select::min, cub::detail::topk::select::max);

  // Generate segment size
  constexpr segment_size_t min_segment_size = 1;
  constexpr auto max_segment_size = static_cast<segment_size_t>(max_segment_bytes / sizeof(key_t));
  constexpr segment_size_t max_static_k     = ::cuda::std::min(max_num_k, max_segment_size);
  const segment_size_t segment_size = GENERATE_COPY(values({min_segment_size, segment_size_t{3}, max_segment_size}),
                                                    take(4, random(min_segment_size, max_segment_size)));

  // Skip invalid combinations
  if (segment_size > max_segment_size)
  {
    SKIP("The given segment size may not exceed the maximum segment size, we statically constrained the algorithm on.");
  }

  // Set the k value
  const segment_size_t k = GENERATE_COPY(values({1, segment_size}), take(3, random(1, segment_size)));

  const segment_index_t num_segments = GENERATE_COPY(values({segment_index_t{1}, segment_index_t{3}}),
                                                    take(4, random(1, 10000)));

  // Capture test parameters
  CAPTURE(c2h::type_name<key_t>(),
          c2h::type_name<segment_size_t>(),
          c2h::type_name<segment_size_t>(),
          max_segment_bytes,
          max_num_k,
          segment_size,
          k,
          num_segments,
          direction);

  // Prepare input & output
  c2h::device_vector<key_t> keys_in_buffer(num_segments * segment_size);
  c2h::device_vector<key_t> keys_out_buffer(num_segments * k, thrust::no_init);
  const int num_key_seeds = 1;
  c2h::gen(C2H_SEED(num_key_seeds), keys_in_buffer);
  auto d_keys_in_ptr   = thrust::raw_pointer_cast(keys_in_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(keys_out_buffer.data());
  auto d_keys_in  = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), k);

  // Copy input for verification
  c2h::device_vector<key_t> expected_keys(keys_in_buffer);

  // Run the top-k algorithm
  dispatch_segmented_topk_keys(
    thrust::raw_pointer_cast(keys_in_buffer.data()),
    thrust::raw_pointer_cast(keys_out_buffer.data()),
    cub::detail::topk::segment_size_uniform<max_segment_size>{segment_size},
    cub::detail::topk::k_uniform<max_static_k>{k},
    cub::detail::topk::select_direction_uniform{direction},
    num_segments,
    cub::detail::topk::total_num_items_guarantee{num_segments * segment_size});

  // Prepare expected results
  if(direction == cub::detail::topk::select::min)
  {
    using comparator_t = cuda::std::less<>;
    segmented_sort_keys(expected_keys, num_segments, segment_size, k, comparator_t{});
  }
  else
  {
    using comparator_t = cuda::std::greater<>;
    segmented_sort_keys(expected_keys, num_segments, segment_size, k, comparator_t{});
  }
  compact_sorted_keys_to_topk(expected_keys, num_segments, segment_size, k);

  REQUIRE(expected_keys == keys_out_buffer);
}
