// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tile DSL kernels backing cub::DeviceTransform's tile path. The kernels
// assume 16-byte alignment on every pointer and 16-byte divisibility on
// num_items so the compiler can pick LDG.E.128. Callers in the dispatch
// header are responsible for honoring those preconditions.

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

#  include <cuda_tile.h>

#  include <cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

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

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CTK_AT_LEAST(13, 3)
