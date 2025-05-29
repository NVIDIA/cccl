/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
 * disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 * following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
 * products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************************************/

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
