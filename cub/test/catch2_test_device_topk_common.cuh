// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_copy.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh> // topk::select::{min, max}

#include <thrust/remove.h>

#include <cuda/iterator>
#include <cuda/std/limits>

#include <c2h/catch2_test_helper.h>

// Function object to generate monotonically non-decreasing values for small key types
template <typename T>
struct inc_t
{
  size_t num_item;
  double value_increment;

  // Needs to be default constructible to qualify as forward iterator
  inc_t() = default;

  inc_t(size_t num_item)
      : num_item(num_item)
  {
    if (num_item < cuda::std::numeric_limits<T>::max())
    {
      value_increment = 1;
    }
    else
    {
      value_increment = static_cast<double>(cuda::std::numeric_limits<T>::max()) / num_item;
    }
  }

  template <typename IndexT>
  __host__ __device__ T operator()(IndexT x)
  {
    return static_cast<T>(value_increment * x);
  }
};

template <typename OffsetItT>
struct segment_size_op
{
  OffsetItT d_offsets;

  template <typename IndexT>
  __host__ __device__ __forceinline__ auto operator()(IndexT segment_id) const
  {
    return d_offsets[segment_id + 1] - d_offsets[segment_id];
  }
};

template <typename OffsetItT, typename KSizesItT>
struct get_output_size_op
{
  OffsetItT offset_it;
  KSizesItT k_it;

  __device__ __forceinline__ cuda::std::int64_t operator()(cuda::std::int64_t segment_id) const
  {
    const auto segment_size = offset_it[segment_id + 1] - offset_it[segment_id];
    return (cuda::std::min) (static_cast<cuda::std::int64_t>(k_it[segment_id]), segment_size);
  }
};

template <typename OffsetItT, typename KSizesItT>
get_output_size_op(OffsetItT, KSizesItT) -> get_output_size_op<OffsetItT, KSizesItT>;

template <typename IteratorT, typename OffsetItT>
struct offset_iterator_op
{
  IteratorT base_it;
  OffsetItT offset_it;

  offset_iterator_op(IteratorT base_it, OffsetItT offset_it)
      : base_it(base_it)
      , offset_it(offset_it)
  {}

  template <typename IndexT>
  __device__ __forceinline__ IteratorT operator()(IndexT segment_id) const
  {
    return base_it + offset_it[segment_id];
  }
};

template <cub::detail::topk::select SelectDirection>
using direction_to_comparator_t =
  cuda::std::conditional_t<SelectDirection == cub::detail::topk::select::min, cuda::std::less<>, cuda::std::greater<>>;

// Function object that maintains two bit-flags:
// (1) one to keep track of the unique items encountered
// (2) another to keep track of the indices where items were written
struct set_bit_flag_for_write_op
{
  std::uint32_t* d_element_flags;
  std::uint32_t* d_index_flags;

  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);

  template <typename OffsetT>
  __host__ __device__ void set_bit_flag(std::uint32_t* d_flags, OffsetT index)
  {
    // Set the n-th bit from a given flags array
    OffsetT uint_index     = index / static_cast<OffsetT>(bits_per_element);
    std::uint32_t bit_flag = 0x00000001U << (index % bits_per_element);
    atomicOr(&d_flags[uint_index], bit_flag);
  }

  template <typename OffsetT, typename T>
  __host__ __device__ void operator()(OffsetT index, T val)
  {
    static_assert(cuda::std::is_integral<T>::value, "set_bit_for_element_op requires values to be of integral type");
    set_bit_flag(d_element_flags, static_cast<OffsetT>(val));
    set_bit_flag(d_index_flags, index);
  }
};

// Helper class to check whether every element from 0...num_elements-1 has been written to the output at some index
// and, similarly, whether every index from 0...num_elements-1 has been written to at least once.
// The first is to ensure that all expected elements have been outputted (regardless of order) and the second is to
// ensure that no output index has been skipped.
class check_unordered_output_helper
{
  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);

  // Boolean flags to indicate whether the correct result has been written
  c2h::device_vector<std::uint32_t> element_flags;
  c2h::device_vector<std::uint32_t> index_flags;
  std::size_t num_elements;

  // Checks whether all results have been written correctly
  void check_bit_flags(const c2h::device_vector<std::uint32_t>& flag_vector)
  {
    auto correctness_flags_end = flag_vector.cbegin() + (num_elements / bits_per_element);
    const bool all_correct =
      thrust::equal(flag_vector.cbegin(), correctness_flags_end, cuda::constant_iterator(0xFFFFFFFFU));

    if (!all_correct)
    {
      using thrust::placeholders::_1;
      auto mismatch_it = thrust::find_if_not(flag_vector.cbegin(), correctness_flags_end, _1 == 0xFFFFFFFFU);
      // Sanity check: if thrust::equals previously "failed", then mismatch_it must not be the end iterator
      REQUIRE(mismatch_it != correctness_flags_end);
      std::uint32_t mismatch_value = *mismatch_it;
      auto bit_index               = 0;
      // Find the first bit that is not set in the mismatch_value
      for (std::uint32_t i = 0; i < bits_per_element; ++i)
      {
        if (((mismatch_value >> i) & 0x01u) == 0)
        {
          bit_index = i;
          break;
        }
      }
      const auto wrong_element_index = (mismatch_it - flag_vector.cbegin()) * bits_per_element + bit_index;
      FAIL("First wrong output index: " << wrong_element_index);
    }
    if (num_elements % bits_per_element != 0)
    {
      auto const last_element_flags = flag_vector[num_elements / bits_per_element];
      for (std::uint32_t i = 0; i < (num_elements % bits_per_element); ++i)
      {
        const auto element_index = (num_elements / bits_per_element) * bits_per_element + i;
        INFO("First wrong output index: " << element_index);
        REQUIRE(((last_element_flags >> i) & 0x01u) == 0x01u);
      }
    }
  }

public:
  // Prepare the helper for a given number of output elements
  check_unordered_output_helper(std::size_t num_elements)
      : num_elements(num_elements)
  {
    element_flags.resize(cuda::ceil_div(num_elements, bits_per_element), 0);
    index_flags.resize(cuda::ceil_div(num_elements, bits_per_element), 0);
  }

  // Prepares and returns a tabulate_output_iterator that checks whether the correct result has been written at each
  // index
  cuda::tabulate_output_iterator<set_bit_flag_for_write_op> get_flagging_output_iterator()
  {
    auto check_op = set_bit_flag_for_write_op{
      thrust::raw_pointer_cast(element_flags.data()), thrust::raw_pointer_cast(index_flags.data())};
    return cuda::make_tabulate_output_iterator(check_op);
  }

  // Checks whether all results have been written correctly
  void check_all_results_correct()
  {
    INFO("Checking whether all of the expected elements were written");
    check_bit_flags(element_flags);
    INFO("Checking whether all of the expected indexes were written");
    check_bit_flags(index_flags);
  }
};

// Function object used to remove all elements outside the top-k within each segment
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

// Stream-compacts each segment to only contain the top-k elements
template <typename KeyT>
void compact_sorted_keys_to_topk(
  c2h::device_vector<KeyT>& d_keys_in, cuda::std::int64_t segment_size, cuda::std::int64_t k)
{
  // Remove all elements within each segment that are not amongst the top-k
  auto new_end = thrust::remove_if(
    d_keys_in.begin(), d_keys_in.end(), cuda::make_counting_iterator(0), remove_out_of_topk_op{segment_size, k});

  // Resize input to new size
  d_keys_in.resize(new_end - d_keys_in.begin());
}

// Stream-compacts each segment to only contain the top-k elements
template <typename KeyT, typename OffsetT>
c2h::device_vector<KeyT> compact_to_topk_batched(
  c2h::device_vector<KeyT>& d_keys_in, const c2h::device_vector<OffsetT>& d_offsets, cuda::std::int64_t k)
{
  // Expect
  const auto num_segments = d_offsets.size() - 1;

  // Maps segments to source pointers: d_keys_in.data() + offset[i]
  auto src_ptrs_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(0), offset_iterator_op{d_keys_in.cbegin(), d_offsets.cbegin()});

  // Calculates the output sizes (if segment size is smaller than k, then output size is segment size, otherwise k)
  auto copy_sizes_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(0), get_output_size_op{d_offsets.cbegin(), cuda::constant_iterator(k)});

  // Calculate destination offsets via prefix sum
  c2h::device_vector<OffsetT> d_output_offsets(num_segments + 1, thrust::no_init);
  thrust::exclusive_scan(copy_sizes_it, copy_sizes_it + num_segments + 1, d_output_offsets.begin());

  OffsetT total_compacted_size = d_output_offsets.back();
  c2h::device_vector<KeyT> d_keys_out(total_compacted_size, thrust::no_init);

  // Map segments to destination pointers: d_keys_out.data() + new_offset[i]
  auto dst_ptrs_it = cuda::make_transform_iterator(
    cuda::make_counting_iterator(0), offset_iterator_op{d_keys_out.begin(), d_output_offsets.cbegin()});

  // Query temporary storage size
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, src_ptrs_it, dst_ptrs_it, copy_sizes_it, num_segments);
  c2h::device_vector<cuda::std::uint8_t> d_temp(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(d_temp.data());

  // Run batched copy to compact top-k elements of each segment to the front of the input buffer
  cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, src_ptrs_it, dst_ptrs_it, copy_sizes_it, num_segments);

  return d_keys_out;
}

template <typename KeyT, typename OffsetItT>
void segmented_sort_keys(
  c2h::device_vector<KeyT>& d_keys_in,
  cuda::std::int64_t num_segments,
  OffsetItT d_segment_offsets_begin_it,
  OffsetItT d_segment_offsets_end_it,
  cub::detail::topk::select direction)
{
  cuda::std::int64_t num_items = d_keys_in.size();

  // Prepare alternate buffer for double buffering
  c2h::device_vector<KeyT> d_keys_alt(num_items, thrust::no_init);
  cub::DoubleBuffer<KeyT> d_keys(
    thrust::raw_pointer_cast(d_keys_in.data()), thrust::raw_pointer_cast(d_keys_alt.data()));

  // Query temporary storage size
  size_t temp_storage_bytes = 0;
  if (direction == cub::detail::topk::select::min)
  {
    cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_segment_offsets_begin_it,
      d_segment_offsets_end_it);

    // Allocate temporary storage
    c2h::device_vector<cuda::std::uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

    // Run segmented sort
    cub::DeviceSegmentedSort::SortKeys(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_segment_offsets_begin_it,
      d_segment_offsets_end_it);
  }
  else
  {
    cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr,
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_segment_offsets_begin_it,
      d_segment_offsets_end_it);

    // Allocate temporary storage
    c2h::device_vector<cuda::std::uint8_t> d_temp_storage(temp_storage_bytes, thrust::no_init);

    // Run segmented sort
    cub::DeviceSegmentedSort::SortKeysDescending(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      d_segment_offsets_begin_it,
      d_segment_offsets_end_it);
  }

  // Make sure the result is returned in the original buffer
  if (d_keys.Current() != thrust::raw_pointer_cast(d_keys_in.data()))
  {
    thrust::copy(d_keys.Current(), d_keys.Current() + num_items, d_keys_in.begin());
  }
}

template <typename KeyT>
void fixed_size_segmented_sort_keys(
  c2h::device_vector<KeyT>& d_keys_in,
  cuda::std::int64_t num_segments,
  cuda::std::int64_t segment_size,
  cub::detail::topk::select direction)
{
  auto segment_offsets_it =
    cuda::make_strided_iterator(cuda::make_counting_iterator<cuda::std::int64_t>(0), segment_size);

  // We materialize the offsets to reduce the number of kernel template specializations
  c2h::device_vector<cuda::std::int64_t> d_segment_offsets(num_segments + 1);
  thrust::copy(segment_offsets_it, segment_offsets_it + (num_segments + 1), d_segment_offsets.begin());

  // Perform segmented sort
  auto d_segment_offsets_begin_it = d_segment_offsets.cbegin();
  auto d_segment_offsets_end_it   = d_segment_offsets_begin_it + 1;
  segmented_sort_keys(d_keys_in, num_segments, d_segment_offsets_begin_it, d_segment_offsets_end_it, direction);
}
