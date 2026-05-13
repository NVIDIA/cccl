// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! `cub::DeviceFind::UpperBound` — haystack sorted, needles unsorted (Thrust vectorized_search parity).

#include <cub/device/device_find.cuh>

#include <cstdint>

#include <nvbench_helper.cuh>

#include "find_bound_common.cuh"

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  bounds_bench_data<T> s(state);

  state.add_element_count(s.needles);
  state.add_global_memory_reads<T>(s.needles);
  state.add_global_memory_writes<std::ptrdiff_t>(s.needles);

  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes{};
  less_t comp{};

  cub::DeviceFind::UpperBound(
    d_temp_storage,
    temp_storage_bytes,
    s.range_ptr(),
    static_cast<std::int64_t>(s.elements),
    s.values_ptr(),
    static_cast<std::int64_t>(s.needles),
    s.output_ptr(),
    comp,
    0);

  thrust::device_vector<nvbench::uint8_t> temp_storage(temp_storage_bytes, thrust::no_init);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) {
               cub::DeviceFind::UpperBound(
                 d_temp_storage,
                 temp_storage_bytes,
                 s.range_ptr(),
                 static_cast<std::int64_t>(s.elements),
                 s.values_ptr(),
                 static_cast<std::int64_t>(s.needles),
                 s.output_ptr(),
                 comp,
                 launch.get_stream());
             });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(bounds_value_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(16, 28, 4))
  .add_int64_axis("NeedlesRatio", {1, 25, 50});
