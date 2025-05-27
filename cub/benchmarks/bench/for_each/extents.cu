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

template <class T>
struct op_t
{
  int* d_count{};

  __device__ void operator()(T, T x, T y, T z, T w) const
  {
    if (x + y + z + w + 1 == T{})
    {
      atomicAdd(d_count, 1);
    }
  }
};

template <class OffsetT>
void for_each_in_extents(nvbench::state& state, nvbench::type_list<OffsetT>)
{
  const auto elements = static_cast<OffsetT>(state.get_int64("Elements{io}"));
  // `d_out` exists for visibility
  // All inputs are equal to `42`, while the operator is searching for `0`.
  // If the operator finds `0` in the input sequence, it's an issue leading to a segfault.
  int* d_out = nullptr;
  state.add_element_count(elements);
  op_t<int> op{d_out};

  size_t temp_size{};
  auto elements_1D = ::cuda::isqrt(::cuda::isqrt(elements));
  cuda::std::dextents<OffsetT, 4> ext{elements_1D, elements_1D, elements_1D, elements_1D};
  cub::DeviceFor::ForEachInExtents(nullptr, temp_size, ext, op);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::DeviceFor::ForEachInExtents(temp_storage, temp_size, ext, op, launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(for_each_in_extents, NVBENCH_TYPE_AXES(offset_types))
  .set_name("base")
  .set_type_axes_names({"OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
