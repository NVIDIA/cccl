// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile port of cub::DeviceTransform. The public surface mirrors
// cub::DeviceTransform::{Transform, Fill}; the kernels are written against the
// tile DSL (cuda::tiles). This header is only safe to include when nvcc is
// invoked with --enable-tile and CTK >= 13.3.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/cmath>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstdint>

namespace cub_tile::detail
{

constexpr int min_bytes_in_flight_per_sm(int cc_x10)
{
  if (cc_x10 >= 1000)
  {
    return 64 * 1024; // B200
  }
  if (cc_x10 >= 900)
  {
    return 48 * 1024; // H100/H200
  }
  if (cc_x10 >= 800)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024;
}

constexpr int min_size(int a)
{
  return a;
}
template <class... Ts>
constexpr int min_size(int a, int b, Ts... rest)
{
  int m = a < b ? a : b;
  return min_size(m, rest...);
}

// mufu_heavy=true tells the policy the functor body has heavy MUFU usage.
// for small data types, vectorized load will make them arrive packed in
// registers and the compiler unpacks them and packs them back. reducing the
// compute work per thread helps here. need profiling to know the exact cause.
template <typename Out, typename... Ins>
constexpr int pick_tile_size(bool mufu_heavy = false, int cc_x10 = 1000)
{
  constexpr int threads_per_block    = 128;
  constexpr int vector_bytes         = 16; // LDG.E.128 -> 16 bytes
  constexpr int max_items_per_thread = 32;
  constexpr int max_occupancy        = 16;

  constexpr int min_elem      = min_size(int(sizeof(Out)), int(sizeof(Ins))...);
  constexpr int items_for_vec = static_cast<int>(::cuda::ceil_div(vector_bytes, min_elem));

  // Fill (zero inputs) keeps the same latency target by counting output bytes.
  constexpr int bytes_per_iter = (sizeof...(Ins) > 0) ? (int(sizeof(Ins)) + ... + 0) : int(sizeof(Out));
  const int target             = min_bytes_in_flight_per_sm(cc_x10);
  const int items_for_latency =
    static_cast<int>(::cuda::ceil_div(target, max_occupancy * threads_per_block * bytes_per_iter));

  int items = items_for_vec > items_for_latency ? items_for_vec : items_for_latency;
  items     = static_cast<int>(::cuda::next_power_of_two(static_cast<unsigned int>(items)));
  if (items > max_items_per_thread)
  {
    items = max_items_per_thread;
  }

  if (mufu_heavy && min_elem < 4)
  {
    const int byte_cap = vector_bytes / min_elem; // 16 for I8, 8 for I16/half/bf16
    if (items > byte_cap)
    {
      items = byte_cap;
    }
  }

  return items * threads_per_block;
}

template <int TileSize, typename Fn, typename Out, typename... Ins>
__tile_global__ void
transform_kernel(int64_t num_items_, Out* __restrict__ out_, const Ins* __restrict__... ins_)
{
  namespace ct = cuda::tiles;

  const auto bx = ct::bid().x;
  Fn fn{};

  auto num_items = ct::assume_bounded_below<0>(ct::assume_divisible<16>(num_items_));
  auto out       = ct::assume_aligned<16>(out_);

  auto out_span = ct::tensor_span{out, ct::extents{num_items}};
  auto out_view = ct::partition_view{out_span, ct::shape<TileSize>{}};

  auto load_one = [bx, num_items](auto* ptr_) {
    auto ptr  = ct::assume_aligned<16>(ptr_);
    auto span = ct::tensor_span{ptr, ct::extents{num_items}};
    auto view = ct::partition_view{span, ct::shape<TileSize>{}};
    return view.load_masked(bx);
  };

  out_view.store_masked(fn(load_one(ins_)...), bx);
}

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

template <int TileSize, typename T>
__tile_global__ void fill_kernel(int64_t num_items_, T* __restrict__ out_, T value)
{
  namespace ct  = cuda::tiles;
  const auto bx = ct::bid().x;

  auto num_items = ct::assume_bounded_below<0>(ct::assume_divisible<16>(num_items_));
  auto out       = ct::assume_aligned<16>(out_);

  auto out_span = ct::tensor_span{out, ct::extents{num_items}};
  auto out_view = ct::partition_view{out_span, ct::shape<TileSize>{}};
  using tile_t  = ct::tile<T, ct::shape<TileSize>>;
  out_view.store_masked(ct::full<tile_t>(value), bx);
}

} // namespace cub_tile::detail

namespace cub_tile
{

struct DeviceTransform
{
  template <int TileSize = 0, bool MufuHeavy = false, typename Fn, typename Out, typename... Ins>
  static cudaError_t
  Transform(::cuda::std::tuple<Ins*...> inputs, Out* output, int64_t num_items, Fn, cudaStream_t stream = 0)
  {
    constexpr int chosen = (TileSize > 0) ? TileSize : detail::pick_tile_size<Out, Ins...>(MufuHeavy);
    return detail::launch_impl<chosen, Fn>(
      inputs, output, num_items, stream, ::cuda::std::index_sequence_for<Ins...>{});
  }

  // Fill
  template <int TileSize = 0, typename T>
  static cudaError_t Fill(T* output, int64_t num_items, T value, cudaStream_t stream = 0)
  {
    if (num_items <= 0)
    {
      return cudaSuccess;
    }
    constexpr int chosen     = (TileSize > 0) ? TileSize : detail::pick_tile_size<T>();
    const int64_t num_blocks = (num_items + chosen - 1) / chosen;
    detail::fill_kernel<chosen, T>
      <<<static_cast<unsigned int>(num_blocks), 1, 0, stream>>>(num_items, output, value);
    return cudaGetLastError();
  }
};

} // namespace cub_tile
