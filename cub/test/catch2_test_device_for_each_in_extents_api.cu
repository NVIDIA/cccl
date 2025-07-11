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

// Allow nested NVTX ranges for this algorithm
// #include "insert_nested_NVTX_range_guard.h"

#include <cub/config.cuh>

// MSVC doesn't support __device__ lambdas
#if !_CCCL_COMPILER(MSVC)

#  include <cub/device/device_for.cuh>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/device_vector.h>
#  include <thrust/fill.h>
#  include <thrust/host_vector.h>

#  include <cuda/std/array>
#  include <cuda/std/mdspan>
#  include <cuda/std/span>

#  include <c2h/catch2_test_helper.h>

// example-begin for-each-in-extents-op
struct linear_store_3D
{
  using data_t = cuda::std::array<int, 3>;

  cuda::std::span<data_t> d_output_raw;

  __device__ void operator()(int idx, int x, int y, int z)
  {
    d_output_raw[idx] = {x, y, z};
  }
};
// example-end for-each-in-extents-op

// clang-format off
C2H_TEST("Device ForEachInExtents", "[ForEachInExtents][device]")
{
  // example-begin for-each-in-extents-example
  using                            data_t = cuda::std::array<int, 3>;
  cuda::std::extents<int, 3, 2, 2> extents{};
  c2h::device_vector<data_t>    d_output(cub::detail::size(extents));
  c2h::host_vector<data_t>      h_output(cub::detail::size(extents));
  auto                             d_output_raw = cuda::std::span<data_t>{thrust::raw_pointer_cast(d_output.data()),
                                                                          3 * 2 * 2};
  c2h::host_vector<data_t> expected = {{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                          {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
                                          {2, 0, 0}, {2, 0, 1}, {2, 1, 0}, {2, 1, 1}};

  cub::DeviceFor::ForEachInExtents(extents, [=] __device__ (int idx, int x, int y, int z) {
    d_output_raw[idx] = {x, y, z};
  });
  h_output = d_output;
  REQUIRE(h_output == expected);

  thrust::fill(d_output.begin(), d_output.end(), data_t{});
  cub::DeviceFor::ForEachInExtents(extents, linear_store_3D{d_output_raw});

  h_output = d_output;
  REQUIRE(h_output == expected);
  // example-end for-each-in-extents-example
}
// clang-format on

#endif // !_CCCL_COMPILER(MSVC)
