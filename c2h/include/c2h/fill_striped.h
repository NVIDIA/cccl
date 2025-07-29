/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda/std/type_traits>

template <typename VectorT, typename = void>
struct scalar_to_vec_t
{
  template <typename T>
  __host__ __device__ __forceinline__ auto operator()(T scalar) const -> VectorT
  {
    return static_cast<VectorT>(scalar);
  }
};

template <typename VectorT>
struct scalar_to_vec_t<VectorT, ::cuda::std::void_t<decltype(VectorT::x)>>
{
  template <typename T>
  __host__ __device__ __forceinline__ auto operator()(T scalar) const -> VectorT
  {
    const auto c = static_cast<decltype(VectorT::x)>(scalar);
    VectorT r;
    constexpr auto components = ::cuda::std::tuple_size_v<VectorT>;
    if constexpr (components >= 1)
    {
      r.x = c;
    }
    if constexpr (components >= 2)
    {
      r.y = c;
    }
    if constexpr (components >= 3)
    {
      r.z = c;
    }
    if constexpr (components >= 4)
    {
      r.w = c;
    }
    return r;
  }
};

template <int LogicalWarpThreads, int ItemsPerThread, int BlockThreads, typename IteratorT>
void fill_striped(IteratorT it)
{
  using T = cub::detail::it_value_t<IteratorT>;

  constexpr int warps_in_block = BlockThreads / LogicalWarpThreads;
  constexpr int items_per_warp = LogicalWarpThreads * ItemsPerThread;
  scalar_to_vec_t<T> convert;

  for (int warp_id = 0; warp_id < warps_in_block; warp_id++)
  {
    const int warp_offset_val = items_per_warp * warp_id;

    for (int lane_id = 0; lane_id < LogicalWarpThreads; lane_id++)
    {
      const int lane_offset = warp_offset_val + lane_id;

      for (int item = 0; item < ItemsPerThread; item++)
      {
        *(it++) = convert(lane_offset + item * LogicalWarpThreads);
      }
    }
  }
}
