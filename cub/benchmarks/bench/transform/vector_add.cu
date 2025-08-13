// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "common.h"

template <typename T, typename OffsetT>
static void vector_add(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto n               = narrow<OffsetT>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> a = generate(n);
  thrust::device_vector<T> b = generate(n);

  thrust::device_vector<T> c = [&]() {
    if constexpr (::cuda::std::is_trivially_constructible_v<T>)
    {
      return thrust::device_vector<T>(n, thrust::no_init);
    }
    else
    {
      return thrust::device_vector<T>(n);
    }
  }();

  state.add_element_count(n);
  state.add_global_memory_reads<T>(2 * n);
  state.add_global_memory_writes<T>(n);

  bench_transform(state, cuda::std::tuple{a.begin(), b.begin()}, c.begin(), n, cuda::std::plus<T>{});
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = all_types;
#endif

NVBENCH_BENCH_TYPES(vector_add, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("vector_add")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
