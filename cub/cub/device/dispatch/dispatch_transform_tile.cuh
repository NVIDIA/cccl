// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile port of cub::DeviceTransform. The public surface mirrors
// cub::DeviceTransform::{Transform, Fill}; the kernels are written against the
// tile DSL (cuda::tiles). This header requires CTK 13.3 or newer and nvcc
// invoked with --enable-tile.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CTK_AT_LEAST(13, 3)

#  include <cub/device/dispatch/kernels/kernel_transform_tile.cuh>
#  include <cub/device/dispatch/tuning/tuning_transform_tile.cuh>

#  include <cuda/std/tuple>
#  include <cuda/std/utility>

#  include <cuda_runtime.h>

#  include <cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

template <int TileSize, typename Fn, typename Out, typename... Ins, ::cuda::std::size_t... Idx>
cudaError_t launch_impl(
  ::cuda::std::tuple<Ins*...> inputs,
  Out* output,
  int64_t num_items,
  cudaStream_t stream,
  ::cuda::std::index_sequence<Idx...>)
{
  if (num_items <= 0)
  {
    return cudaSuccess;
  }

  const int64_t num_blocks = (num_items + TileSize - 1) / TileSize;

  transform_kernel<TileSize, Fn><<<static_cast<unsigned int>(num_blocks), 1, 0, stream>>>(
    num_items, output, ::cuda::std::get<Idx>(inputs)...);

  return cudaGetLastError();
}

struct DeviceTransform
{
  template <int TileSize = 0, bool MufuHeavy = false, typename Fn, typename Out, typename... Ins>
  static cudaError_t
  Transform(::cuda::std::tuple<Ins*...> inputs, Out* output, int64_t num_items, Fn, cudaStream_t stream = 0)
  {
    constexpr int chosen = (TileSize > 0) ? TileSize : pick_tile_size<Out, Ins...>(MufuHeavy);
    return launch_impl<chosen, Fn>(inputs, output, num_items, stream, ::cuda::std::index_sequence_for<Ins...>{});
  }

  // Fill
  template <int TileSize = 0, typename T>
  static cudaError_t Fill(T* output, int64_t num_items, T value, cudaStream_t stream = 0)
  {
    if (num_items <= 0)
    {
      return cudaSuccess;
    }
    constexpr int chosen     = (TileSize > 0) ? TileSize : pick_tile_size<T>();
    const int64_t num_blocks = (num_items + chosen - 1) / chosen;
    fill_kernel<chosen, T><<<static_cast<unsigned int>(num_blocks), 1, 0, stream>>>(num_items, output, value);
    return cudaGetLastError();
  }
};

} // namespace detail::transform::tile

CUB_NAMESPACE_END

// Compatibility shim. Existing benches and tests still call
// cub_tile::DeviceTransform; once they move to cub::DeviceTransform with named
// functors and the trait dispatch, this alias can be removed.
namespace cub_tile
{
using DeviceTransform = ::cub::detail::transform::tile::DeviceTransform;
} // namespace cub_tile

#endif // _CCCL_CTK_AT_LEAST(13, 3)
