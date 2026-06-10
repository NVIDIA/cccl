// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_type.cuh>

#include <thrust/equal.h>

#include <cuda/iterator>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/limits>

#include <c2h/catch2_test_helper.h>

namespace detail
{
// Helper that concatenates two iterators into a single iterator
template <typename FirstSegmentItT, typename SecondSegmentItT>
struct concat_iterators_op
{
  FirstSegmentItT first_it;
  SecondSegmentItT second_it;
  ::cuda::std::int64_t num_first_items;

  __host__ __device__ _CCCL_FORCEINLINE auto operator()(::cuda::std::int64_t i)
  {
    if (i < num_first_items)
    {
      return first_it[i];
    }
    else
    {
      return second_it[i - num_first_items];
    }
  }
};

template <typename FirstSegmentItT, typename SecondSegmentItT>
auto make_concat_iterators_op(FirstSegmentItT first_it, SecondSegmentItT second_it, ::cuda::std::int64_t num_first_items)
{
  return cuda::make_transform_iterator(
    cuda::counting_iterator(::cuda::std::int64_t{0}),
    concat_iterators_op<FirstSegmentItT, SecondSegmentItT>{first_it, second_it, num_first_items});
}

template <typename ExpectedValuesItT>
struct flag_correct_writes_op
{
  ExpectedValuesItT expected_it;
  std::uint32_t* d_correctness_flags;

  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);
  template <typename OffsetT, typename T>
  __host__ __device__ void operator()(OffsetT index, T val)
  {
    // Set bit-flag if the correct result has been written at the given index
    if (expected_it[index] == val)
    {
      OffsetT uint_index     = index / static_cast<OffsetT>(bits_per_element);
      std::uint32_t bit_flag = 0x00000001U << (index % bits_per_element);
      atomicOr(&d_correctness_flags[uint_index], bit_flag);
    }
  }
};

template <typename ExpectedValuesItT>
flag_correct_writes_op<ExpectedValuesItT> static make_checking_write_op(
  ExpectedValuesItT expected_it, std::uint32_t* d_correctness_flags)
{
  return flag_correct_writes_op<ExpectedValuesItT>{expected_it, d_correctness_flags};
}

// Struct to help verify results for large problem sizes in a memory-efficient way:
// We use a tabulate iterator that, whenever the algorithm-under-test writes an item, checks whether that item
// corresponds to the expected value at that index and, if correct, sets a boolean flag at that index.
struct large_problem_test_helper
{
  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);

  // Boolean flags to indicate whether the correct result has been written at each index
  c2h::device_vector<std::uint32_t> correctness_flags;
  std::size_t num_elements;

  // Prepare the helper for a given number of output elements
  large_problem_test_helper(std::size_t num_elements)
      : num_elements(num_elements)
  {
    correctness_flags.resize(::cuda::ceil_div(num_elements, bits_per_element), 0);
  }

  // Prepares and returns a tabulate_output_iterator that checks whether the correct result has been written at each
  // index
  template <typename ExpectedValuesItT>
  cuda::tabulate_output_iterator<flag_correct_writes_op<ExpectedValuesItT>>
  get_flagging_output_iterator(ExpectedValuesItT expected_it)
  {
    auto check_op = make_checking_write_op(expected_it, thrust::raw_pointer_cast(correctness_flags.data()));
    return cuda::make_tabulate_output_iterator(check_op);
  }

  // Checks whether all results have been written correctly
  void check_all_results_correct()
  {
    auto correctness_flags_end = correctness_flags.cbegin() + (num_elements / bits_per_element);
    const bool all_correct =
      thrust::equal(correctness_flags.cbegin(), correctness_flags_end, cuda::constant_iterator(0xFFFFFFFFU));

    if (!all_correct)
    {
      using thrust::placeholders::_1;
      auto mismatch_it = thrust::find_if_not(correctness_flags.cbegin(), correctness_flags_end, _1 == 0xFFFFFFFFU);
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
      const auto wrong_element_index = (mismatch_it - correctness_flags.cbegin()) * bits_per_element + bit_index;
      FAIL("First wrong output index: " << wrong_element_index);
    }
    if (num_elements % bits_per_element != 0)
    {
      auto const last_element_flags = correctness_flags[num_elements / bits_per_element];
      for (std::uint32_t i = 0; i < (num_elements % bits_per_element); ++i)
      {
        const auto element_index = (num_elements / bits_per_element) * bits_per_element + i;
        INFO("First wrong output index: " << element_index);
        REQUIRE(((last_element_flags >> i) & 0x01u) == 0x01u);
      }
    }
  }
};

template <typename Offset>
auto make_large_offset(::cuda::std::size_t num_extra_items = 2000000ULL) -> Offset
{
  // Clamp 64-bit offset type problem sizes to just slightly larger than 2^32 items
  const auto num_items_max_ull = ::cuda::std::clamp(
    static_cast<::cuda::std::size_t>(::cuda::std::numeric_limits<Offset>::max()),
    ::cuda::std::size_t{0},
    ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max() + num_extra_items);
  return static_cast<Offset>(num_items_max_ull);
}
} // namespace detail
