// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_for.cuh>

#include <nvbench_helper.cuh>

template <class T>
struct op_t
{
  int* d_count{};

  __device__ void operator()(T val) const
  {
    if (val == T{})
    {
      atomicAdd(d_count, 1);
    }
  }
};

template <class T, class OffsetT>
void for_each(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t  = const T*;
  using output_it_t = int*;
  using offset_t    = OffsetT;

  const auto elements = static_cast<offset_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in(elements, T{42});

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = nullptr;

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);

  op_t<T> op{d_out};

  std::size_t temp_size{};
  cub::DeviceFor::ForEachCopyN(nullptr, temp_size, d_in, elements, op);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFor::ForEachCopyN(temp_storage, temp_size, d_in, elements, op, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(for_each, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
