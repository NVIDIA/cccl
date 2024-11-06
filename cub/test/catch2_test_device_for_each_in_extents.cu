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
#include <cub/config.cuh>

#if _CCCL_STD_VER >= 2017

#  include <cub/device/device_for_each_in_extents.cuh>

#  include <thrust/detail/raw_pointer_cast.h>

#  include <cuda/std/array>

#  include "c2h/catch2_test_helper.cuh"
#  include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceForEachInExtents::ForEachInExtents, device_for_each_in_extents);

/***********************************************************************************************************************
 * UTLILITIES
 **********************************************************************************************************************/

template <typename T, typename IndexType, int Rank = 0, size_t... Extents, typename... IndicesType>
static void fill_linear_impl(
  c2h::host_vector<T>& vector, const cuda::std::extents<IndexType, Extents...>& ext, size_t& pos, IndicesType... indices)
{
  if constexpr (Rank < sizeof...(Extents) /*ext.rank()*/)
  {
    for (IndexType i = 0; i < ext.extent(Rank); ++i)
    {
      fill_linear_impl<T, IndexType, Rank + 1>(vector, ext, pos, indices..., i);
    }
  }
  else
  {
    vector[pos++] = {indices...};
  }
}

template <typename T, typename IndexType, size_t... Extents>
static void fill_linear(c2h::host_vector<T>& vector, const cuda::std::extents<IndexType, Extents...>& ext)
{
  size_t pos = 0;
  fill_linear_impl(vector, ext, pos);
}

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

C2H_TEST("DeviceForEachInExtents 1D static", "[ForEachInExtents][static][device]")
{
  using data_t = cuda::std::array<int, 1>;
  cuda::std::extents<int, 5> ext{};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = thrust::raw_pointer_cast(d_output.data());

  device_for_each_in_extents(ext, [d_output_raw] __device__(auto idx, auto x) {
    d_output_raw[idx] = {x};
  });
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
  REQUIRE(h_output == h_output_gpu);
}

C2H_TEST("DeviceForEachInExtents 3D static", "[ForEachInExtents][static][device]")
{
  using data_t = cuda::std::array<int, 3>;
  cuda::std::extents<int, 5, 3, 4> ext{};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = thrust::raw_pointer_cast(d_output.data());

  device_for_each_in_extents(ext, [d_output_raw] __device__(auto idx, auto x, auto y, auto z) {
    d_output_raw[idx] = {x, y, z};
  });
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
  REQUIRE(h_output == h_output_gpu);
}

C2H_TEST("DeviceForEachInExtents 3D dynamic", "[ForEachInExtents][dynamic][device]")
{
  using data_t = cuda::std::array<int, 3>;
  auto X       = GENERATE_COPY(take(3, random(2, 10)));
  auto Y       = GENERATE_COPY(take(3, random(2, 10)));
  auto Z       = GENERATE_COPY(take(3, random(2, 10)));
  cuda::std::dextents<int, 3> ext{X, Y, Z};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = thrust::raw_pointer_cast(d_output.data());

  device_for_each_in_extents(ext, [d_output_raw] __device__(auto idx, auto x, auto y, auto z) {
    d_output_raw[idx] = {x, y, z};
  });
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
  REQUIRE(h_output == h_output_gpu);
}

C2H_TEST("DeviceForEachInExtents 4D static", "[ForEachInExtents][static][device]")
{
  using data_t = cuda::std::array<int, 4>;
  cuda::std::extents<int, 3, 2, 6, 5> ext{};
  c2h::device_vector<data_t> d_output(cub::detail::size(ext), data_t{});
  c2h::host_vector<data_t> h_output(cub::detail::size(ext), data_t{});
  auto d_output_raw = thrust::raw_pointer_cast(d_output.data());

  device_for_each_in_extents(ext, [d_output_raw] __device__(auto idx, auto x, auto y, auto z, auto w) {
    d_output_raw[idx] = {x, y, z, w};
  });
  c2h::host_vector<data_t> h_output_gpu = d_output;
  fill_linear(h_output, ext);
  REQUIRE(h_output == h_output_gpu);
}

#endif // _CCCL_STD_VER >= 2017
