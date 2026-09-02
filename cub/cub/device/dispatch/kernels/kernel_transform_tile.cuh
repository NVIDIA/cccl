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

#  include <cuda/std/cstdint>

#  include <nv/target>

#  include <cuda_tile.h>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{
// Build a tile partition_view for a 1D contiguous buffer. The two annotations are load-bearing:
//   assume_aligned<16>      -- promises the pointer is 16-byte aligned, so the compiler can pick LDG.E.128 vectorized
//                              loads/stores.
//   ct::extents<int64_t,..> -- explicit element type on the extent; CTAD would deduce uint32_t and wrap at 2^32.
//                              int64_t lets us cover the full num_items range.
// The caller is responsible for honoring assume_aligned<16>; the dispatch header's runtime_preconditions_valid
// enforces this before launching either kernel.
template <int TileSize, typename T, typename N>
[[nodiscard]] __tile__ auto make_aligned_partition_view(T* ptr, N n)
{
  namespace ct         = ::cuda::tiles;
  const auto ptr_align = ct::assume_aligned<16>(ptr);
  auto span            = ct::tensor_span{ptr_align, ct::extents<::cuda::std::int64_t, ct::dynamic_extent>{n}};
  return ct::partition_view{span, ct::shape<TileSize>{}};
}

// Tile DSL kernel backing cub::DeviceTransform's tile path. It assumes 16-byte pointer alignment + 16-divisible
// num_items (so the compiler picks LDG.E.128); the dispatch header honors that. NV_IF_TARGET(NV_PROVIDES_SM_80)
// guards the body -- tile needs sm_80+, so sub-80 arches get a no-op kernel (dispatch only launches it on sm_80+).
//   assume_divisible<16>     -- num_items % 16 == 0, so the tile DSL can elide tail handling.
//   assume_bounded_below<0>  -- num_items >= 0; enables sign-comparison simplifications.
//
// NOTE: make_aligned_partition_view is invoked directly -- do NOT wrap these calls in a lambda: nvcc 13.4
// miscompiles a templated __tile__ helper called via a lambda under --expt-relaxed-constexpr (invalid IR).
template <int TileSize, typename Fn, typename Out, typename... Ins>
__tile_global__ void transform_kernel(const ::cuda::std::int64_t num_items, Out* out, const Ins*... ins)
{
  namespace ct = ::cuda::tiles;
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (const auto bx = ct::bid().x; const auto n = ct::assume_bounded_below<0>(ct::assume_divisible<16>(num_items));
     const auto out_view                       = detail::transform::tile::make_aligned_partition_view<TileSize>(out, n);
     out_view.store_masked(
       Fn{}(detail::transform::tile::make_aligned_partition_view<TileSize>(ins, n).load_masked(bx)...), bx);));
}
} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
