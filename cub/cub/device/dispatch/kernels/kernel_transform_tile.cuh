// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#include <cub/device/dispatch/dispatch_transform_tile_config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUB_HAS_TILE_TRANSFORM()

#  include <cuda_tile.h>

#  include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

// Build a tile partition_view for a 1D contiguous buffer. The two annotations are load-bearing:
//   assume_aligned<16>      -- promises the pointer is 16-byte aligned, so the compiler can pick
//                              LDG.E.128 vectorized loads/stores.
//   ct::extents<int64_t,..> -- explicit element type on the extent; CTAD would deduce uint32_t and
//                              wrap at 2^32. int64_t lets us cover the full num_items range.
// The caller is responsible for honoring assume_aligned<16>; the dispatch header's
// runtime_preconditions_ok enforces this before launching either kernel.
template <int TileSize, typename T, typename N>
__tile__ auto make_partition_view(T* ptr, N n)
{
  namespace ct        = ::cuda::tiles;
  const auto ptr_align = ct::assume_aligned<16>(ptr);
  auto span            = ct::tensor_span{ptr_align, ct::extents<::cuda::std::int64_t, ct::dynamic_extent>{n}};
  return ct::partition_view{span, ct::shape<TileSize>{}};
}

// Tile DSL kernels backing cub::DeviceTransform's tile path. The kernels assume 16-byte alignment on
// every pointer and 16-byte divisibility on num_items so the compiler can pick LDG.E.128. Callers in
// the dispatch header are responsible for honoring those preconditions.
//
// assume_divisible<16>      -- promises num_items % 16 == 0, so the tile DSL can elide tail handling.
// assume_bounded_below<0>   -- promises num_items >= 0; enables sign-comparison simplifications.
template <int TileSize, typename Fn, typename Out, typename... Ins>
__tile_global__ void
transform_kernel(const ::cuda::std::int64_t num_items, Out* __restrict__ out, const Ins* __restrict__... ins)
{
  namespace ct  = ::cuda::tiles;
  const auto bx = ct::bid().x;
  Fn fn{};

  const auto n     = ct::assume_bounded_below<0>(ct::assume_divisible<16>(num_items));
  auto out_view    = make_partition_view<TileSize>(out, n);
  auto load_one    = [bx, n](auto* ptr) { return make_partition_view<TileSize>(ptr, n).load_masked(bx); };

  out_view.store_masked(fn(load_one(ins)...), bx);
}

template <int TileSize, typename T>
__tile_global__ void fill_kernel(const ::cuda::std::int64_t num_items, T* __restrict__ out, const T value)
{
  namespace ct  = ::cuda::tiles;
  const auto bx = ct::bid().x;

  const auto n  = ct::assume_bounded_below<0>(ct::assume_divisible<16>(num_items));
  auto out_view = make_partition_view<TileSize>(out, n);
  using tile_t  = ct::tile<T, ct::shape<TileSize>>;
  out_view.store_masked(ct::full<tile_t>(value), bx);
}

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
