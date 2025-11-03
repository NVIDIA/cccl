// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements      = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto needles_ratio = static_cast<std::size_t>(state.get_int64("NeedlesRatio"));
  const auto needles       = needles_ratio * static_cast<std::size_t>(static_cast<double>(elements) / 100.0);

  thrust::device_vector<T> data = generate(elements + needles);
  thrust::device_vector<T> result(needles);
  thrust::sort(data.begin(), data.begin() + elements);

  state.add_element_count(needles);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::lower_bound(
                 policy(alloc, launch),
                 data.begin(),
                 data.begin() + elements,
                 data.begin() + elements,
                 data.end(),
                 result.begin());
             });
}

using types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_axis("NeedlesRatio", {1, 25, 50});
