// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "nvbench_helper.cuh"

template <class KeyT, class ValueT>
static void basic(nvbench::state& state, nvbench::type_list<KeyT, ValueT>)
{
  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements"));
  const bit_entropy entropy = str_to_entropy(state.get_string("Entropy"));

  thrust::device_vector<KeyT> in_keys = generate(elements, entropy);
  thrust::device_vector<KeyT> keys(elements);

  thrust::device_vector<ValueT> in_vals = generate(elements);
  thrust::device_vector<ValueT> vals(elements);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<KeyT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               keys = in_keys;
               vals = in_vals;
               timer.start();
               thrust::sort_by_key(policy(alloc, launch), keys.begin(), keys.end(), vals.begin());
               timer.stop();
             });
}

using key_types   = integral_types;
using value_types = integral_types;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.201"});
