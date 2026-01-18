// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh> // topk::select::{min, max}

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
    static_assert(::cuda::std::is_integral<T>::value, "set_bit_for_element_op requires values to be of integral type");
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
    element_flags.resize(::cuda::ceil_div(num_elements, bits_per_element), 0);
    index_flags.resize(::cuda::ceil_div(num_elements, bits_per_element), 0);
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

template <typename KeyT>
void segmented_sort_keys(c2h::device_vector<KeyT>& d_keys_in,
                         cuda::std::int64_t num_segments,
                         cuda::std::int64_t segment_size,
                         cub::detail::topk::select direction)
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
  if (direction == cub::detail::topk::select::min)
  {
    cub::DeviceSegmentedSort::SortKeys(
      nullptr, temp_storage_bytes, d_keys, num_items, num_segments, segment_offsets_it, (segment_offsets_it + 1));

    // Allocate temporary storage
    c2h::device_vector<cuda::std::uint8_t> d_temp_storage(temp_storage_bytes);

    // Run segmented sort
    cub::DeviceSegmentedSort::SortKeys(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      segment_offsets_it,
      (segment_offsets_it + 1));
  }
  else
  {
    cub::DeviceSegmentedSort::SortKeysDescending(
      nullptr, temp_storage_bytes, d_keys, num_items, num_segments, segment_offsets_it, (segment_offsets_it + 1));

    // Allocate temporary storage
    c2h::device_vector<cuda::std::uint8_t> d_temp_storage(temp_storage_bytes);

    // Run segmented sort
    cub::DeviceSegmentedSort::SortKeysDescending(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      d_keys,
      num_items,
      num_segments,
      segment_offsets_it,
      (segment_offsets_it + 1));
  }

  // Make sure the result is returned in the original buffer
  if (d_keys.Current() != thrust::raw_pointer_cast(d_keys_in.data()))
  {
    thrust::copy(d_keys.Current(), d_keys.Current() + num_items, d_keys_in.begin());
  }
}
