// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_type.cuh>

#include <thrust/equal.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>

#include <cuda/std/__cccl/execution_space.h>

#include <c2h/catch2_test_helper.h>

namespace detail
{

template <typename ExpectedValuesItT>
struct flag_correct_writes_op
{
  ExpectedValuesItT expected_it;
  std::uint32_t* d_correctness_flags;

  static constexpr auto bits_per_element = 8 * sizeof(std::uint32_t);
  template <typename OffsetT, typename T>
  _CCCL_HOST_DEVICE void operator()(OffsetT index, T val)
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
  thrust::tabulate_output_iterator<flag_correct_writes_op<ExpectedValuesItT>>
  get_flagging_output_iterator(ExpectedValuesItT expected_it)
  {
    auto check_op = make_checking_write_op(expected_it, thrust::raw_pointer_cast(correctness_flags.data()));
    return thrust::make_tabulate_output_iterator(check_op);
  }

  // Checks whether all results have been written correctly
  void check_all_results_correct()
  {
    REQUIRE(thrust::equal(correctness_flags.cbegin(),
                          correctness_flags.cbegin() + (num_elements / bits_per_element),
                          thrust::make_constant_iterator(0xFFFFFFFFU)));
    if (num_elements % bits_per_element != 0)
    {
      std::uint32_t last_element_flags = (0x00000001U << (num_elements % bits_per_element)) - 0x01U;
      REQUIRE(correctness_flags[num_elements / bits_per_element] == last_element_flags);
    }
  }
};

} // namespace detail
