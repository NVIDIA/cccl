// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "base.cuh"

struct op_t
{
  template <class PolicyT, class InputIterator1, class InputIterator2, class OutputIterator>
  __host__ OutputIterator operator()(
    const PolicyT& policy,
    InputIterator1 first1,
    InputIterator1 last1,
    InputIterator2 first2,
    InputIterator2 last2,
    OutputIterator result) const
  {
    return thrust::set_difference(policy, first1, last1, first2, last2, result);
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T> tl)
{
  basic(state, tl, op_t{});
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"})
  .add_int64_axis("SizeRatio", {25, 50, 75});
