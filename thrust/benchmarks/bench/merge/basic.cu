// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements        = static_cast<std::size_t>(state.get_int64("Elements"));
  const auto size_ratio      = static_cast<std::size_t>(state.get_int64("InputSizeRatio"));
  const auto entropy         = str_to_entropy(state.get_string("Entropy"));
  const auto elements_in_lhs = static_cast<std::size_t>(static_cast<double>(size_ratio * elements) / 100.0);

  thrust::device_vector<T> out(elements);
  thrust::device_vector<T> in = generate(elements, entropy);
  thrust::sort(in.begin(), in.begin() + elements_in_lhs);
  thrust::sort(in.begin() + elements_in_lhs, in.end());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::merge(
                 policy(alloc, launch),
                 in.cbegin(),
                 in.cbegin() + elements_in_lhs,
                 in.cbegin() + elements_in_lhs,
                 in.cend(),
                 out.begin());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"})
  .add_int64_axis("InputSizeRatio", {25, 50, 75});
