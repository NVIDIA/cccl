// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>

#include "nvbench_helper.cuh"

template <class KeyT, class ValueT>
static void basic(nvbench::state& state, nvbench::type_list<KeyT, ValueT>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  constexpr std::size_t min_segment_size = 1;
  const std::size_t max_segment_size     = static_cast<std::size_t>(state.get_int64("MaxSegSize"));

  thrust::device_vector<KeyT> in_keys  = generate.uniform.key_segments(elements, min_segment_size, max_segment_size);
  thrust::device_vector<KeyT> out_keys = in_keys;
  thrust::device_vector<ValueT> in_vals(elements);

  const std::size_t unique_keys =
    ::cuda::std::distance(out_keys.begin(), thrust::unique(out_keys.begin(), out_keys.end()));

  thrust::device_vector<ValueT> out_vals(unique_keys);

  state.add_element_count(elements);
  state.add_global_memory_reads<KeyT>(elements);
  state.add_global_memory_reads<ValueT>(elements);

  state.add_global_memory_writes<KeyT>(unique_keys);
  state.add_global_memory_writes<ValueT>(unique_keys);

  caching_allocator_t alloc;
  state.exec(
    nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      thrust::reduce_by_key(
        policy(alloc, launch), in_keys.begin(), in_keys.end(), in_vals.begin(), out_keys.begin(), out_vals.begin());
    });
}

using key_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t
#if NVBENCH_HELPER_HAS_I128
                     ,
                     int128_t
#endif
                     >;

using value_types = all_types;

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(key_types, value_types))
  .set_name("base")
  .set_type_axes_names({"KeyT{ct}", "ValueT{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegSize", {1, 4, 8});
