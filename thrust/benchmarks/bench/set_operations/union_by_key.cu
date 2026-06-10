// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "by_key.cuh"

struct op_t
{
  static constexpr bool read_all_values = true;

  template <class PolicyT,
            class InputIterator1,
            class InputIterator2,
            class InputIterator3,
            class InputIterator4,
            class OutputIterator1,
            class OutputIterator2>
  __host__ cuda::std::pair<OutputIterator1, OutputIterator2> operator()(
    const PolicyT& policy,
    InputIterator1 keys_first1,
    InputIterator1 keys_last1,
    InputIterator2 keys_first2,
    InputIterator2 keys_last2,
    InputIterator3 values_first1,
    InputIterator4 values_first2,
    OutputIterator1 keys_result,
    OutputIterator2 values_result) const
  {
    return thrust::set_union_by_key(
      policy, keys_first1, keys_last1, keys_first2, keys_last2, values_first1, values_first2, keys_result, values_result);
  }
};

template <class KeyT, class ValueT>
static void basic(nvbench::state& state, nvbench::type_list<KeyT, ValueT> tl)
{
  basic(state, tl, op_t{});
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"})
  .add_int64_axis("SizeRatio", {25, 50, 75});
