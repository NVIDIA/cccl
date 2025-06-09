// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/device_for.cuh>

#include <cuda/cmath>

#include <nvbench_helper.cuh>

template <typename T, typename OffsetT>
struct op_t
{
  using ext_t = cuda::std::dextents<OffsetT, 2>;
  cuda::std::mdspan<T, ext_t> temp_in;
  cuda::std::mdspan<T, ext_t> temp_out;

  __device__ void operator()(OffsetT, OffsetT row, OffsetT column) const
  {
    if (row > 0 && column > 0 && row < temp_in.extent(0) - 1 && column < temp_in.extent(1) - 1)
    {
      T d2tdx2              = temp_in(row, column - 1) - 2 * temp_in(row, column) + temp_in(row, column + 1);
      T d2tdy2              = temp_in(row - 1, column) - 2 * temp_in(row, column) + temp_in(row + 1, column);
      temp_out(row, column) = temp_in(row, column) + 0.2f * (d2tdx2 + d2tdy2);
    }
    else
    {
      temp_out(row, column) = temp_in(row, column);
    }
  }
};

template <class T, class OffsetT>
void for_each_in_extents(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using it_t          = T*;
  using ext_t         = cuda::std::dextents<OffsetT, 2>;
  const auto elements = static_cast<OffsetT>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in(elements, T{42});
  thrust::device_vector<T> out(elements);
  it_t d_in  = thrust::raw_pointer_cast(in.data());
  it_t d_out = thrust::raw_pointer_cast(out.data());
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  auto elements_1D = ::cuda::isqrt(elements);
  ext_t ext{elements_1D, elements_1D};
  cuda::std::mdspan<T, ext_t> temp_in{d_in, ext};
  cuda::std::mdspan<T, ext_t> temp_out{d_out, ext};
  op_t<T, OffsetT> op{temp_in, temp_out};

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFor::ForEachInExtents(ext, op, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(for_each_in_extents, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
