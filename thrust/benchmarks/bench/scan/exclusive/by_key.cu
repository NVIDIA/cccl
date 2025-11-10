// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "nvbench_helper.cuh"

template <class KeyT, class ValueT>
static void scan(nvbench::state& state, nvbench::type_list<KeyT, ValueT>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  thrust::device_vector<ValueT> in_vals(elements);
  thrust::device_vector<ValueT> out_vals(elements);
  thrust::device_vector<KeyT> keys = generate.uniform.key_segments(elements, 0, 5200);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);
  state.add_global_memory_writes<ValueT>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               thrust::exclusive_scan_by_key(
                 policy(alloc, launch), keys.cbegin(), keys.cend(), in_vals.cbegin(), out_vals.begin());
             });
}

using key_types = all_types;
using value_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#if NVBENCH_HELPER_HAS_I128
                     ,
                     int128_t
#endif
                     >;

NVBENCH_BENCH_TYPES(scan, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4));
