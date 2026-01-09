// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
