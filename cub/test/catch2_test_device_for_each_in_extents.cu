/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "insert_nested_NVTX_range_guard.h"
// above header needs to be included first

#include <cub/device/device_for_each_in_extents.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>

#include "c2h/catch2_test_helper.cuh"
#include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceForEachInExtents::ForEachInExtents, device_for_each_in_extents);

C2H_TEST("DDeviceForEachInExtents works", "[ForEachInExtents][device]")
{
  using data_t = cuda::std::tuple<int, int, int>; // REQUIRE(x == y) doesn't work with cuda::std::array
  cuda::std::extents<int, 5, 3, 4> ext{};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = thrust::raw_pointer_cast(d_output.data());

  device_for_each_in_extents(ext, [d_output_raw] __device__(auto x, auto y, auto z) {
    auto id          = threadIdx.x + blockDim.x * blockIdx.x;
    d_output_raw[id] = {x, y, z};
  });
  c2h::host_vector<data_t> h_output_gpu = d_output;

  for (int l = 0, i = 0; i < ext.extent(0); ++i)
  {
    for (int j = 0; j < ext.extent(1); ++j)
    {
      for (int k = 0; k < ext.extent(2); ++k)
      {
        h_output[l++] = {i, j, k};
      }
    }
  }
  REQUIRE(h_output == h_output_gpu);
}
