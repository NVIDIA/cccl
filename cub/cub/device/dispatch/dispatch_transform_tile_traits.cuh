// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-time policy for cub::DeviceTransform's tile path.
//
// PUBLIC EXTENSION POINTS (cub::transform) -- two independent axes:
//   tile_eligible<Op, T, NIn>   -- specialize to true_type to opt a (functor
//                                   type, element type, input arity) combo into
//                                   the tile dispatch path. Eligibility only.
//   tile_eligible_v<...>        -- variable-template companion.
//   tile_operator<Op>           -- the __tile__ functor the tile kernel runs
//                                   for Op. No default: every tile-eligible Op
//                                   must specialize it with `using type = <a
//                                   stateless __tile__ functor mirroring Op>`,
//                                   because a scalar functor (e.g.
//                                   cuda::std::plus<__half>) cannot be invoked
//                                   on ct::tile. Omitting it is a clear
//                                   static_assert, not a cryptic kernel error.
//   tile_operator_t<Op>         -- alias for tile_operator<Op>::type.
//   tile_mufu_heavy<Op>         -- specialize to flag Op as MUFU-heavy; the
//                                   tile policy picker uses this hint.
//   tile_mufu_heavy_v<...>      -- variable-template companion.
//
// Eligibility ("may this combo use the tile path?") and substitution ("which
// __tile__ functor do we actually run?") are separate traits, so an eligible op
// always registers both: tile_eligible<Op,T,NIn> and tile_operator<Op>.
//
// INTERNAL (cub::detail::transform::tile):
//   tile_plus, tile_multiplies   -- shipped tile-friendly substitutes used by
//                                    the built-in specializations below.

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

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda_tile.h>

CUB_NAMESPACE_BEGIN

// Public extension surface.
namespace transform
{
template <typename Op, typename T, ::cuda::std::size_t NIn>
struct tile_eligible : ::cuda::std::false_type
{};

template <typename Op, typename T, ::cuda::std::size_t NIn>
inline constexpr bool tile_eligible_v = tile_eligible<Op, T, NIn>::value;

// The __tile__ functor the tile kernel runs for Op -- the tile-side mirror of the scalar Op. There is
// no default: a scalar functor cannot be invoked on ct::tile, so every tile-eligible Op must specialize
// this with a `type` naming a stateless __tile__ functor. tile_eligible<Op,...> says a combo MAY use the
// tile path; tile_operator<Op> says WHAT the tile kernel runs.
template <typename Op>
struct tile_operator
{
  static_assert(sizeof(Op) == 0,
                "cub::transform::tile_operator<Op> must be specialized for every tile-eligible Op: "
                "provide `using type = <stateless __tile__ functor mirroring Op>`.");
};

template <typename Op>
using tile_operator_t = typename tile_operator<Op>::type;

// Hint that Op uses MUFU (multi-function unit, sin/cos/exp/log/tanh/rcp/rsq). Setting this makes
// the tile policy picker cap items/thread so MUFU pipes are not oversaturated.
template <typename Op>
struct tile_mufu_heavy : ::cuda::std::false_type
{};

template <typename Op>
inline constexpr bool tile_mufu_heavy_v = tile_mufu_heavy<Op>::value;
} // namespace transform

// Internal substitutes shipped by CCCL.
namespace detail::transform::tile
{
// Tile-friendly mirrors of common cuda::std ops. Each has a __tile__
// templated operator() so it can be invoked from inside transform_kernel
// where the arguments are ct::tile<T, ...> rather than scalar T.
struct tile_plus
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a + b;
  }
};

struct tile_multiplies
{
  template <class A, class B>
  __tile__ auto operator()(A a, B b) const
  {
    return a * b;
  }
};
} // namespace detail::transform::tile

// Built-in trait specializations live in the public namespace alongside the
// trait, but reference the internal substitute functors.
namespace transform
{
// cuda::std::plus / multiplies are scalar ops, so each is marked eligible and given a tile_operator mirror.
#  if _CCCL_HAS_NVFP16()
template <>
struct tile_eligible<::cuda::std::plus<::__half>, ::__half, 2> : ::cuda::std::true_type
{};
template <>
struct tile_eligible<::cuda::std::multiplies<::__half>, ::__half, 2> : ::cuda::std::true_type
{};
template <>
struct tile_operator<::cuda::std::plus<::__half>>
{
  using type = cub::detail::transform::tile::tile_plus;
};
template <>
struct tile_operator<::cuda::std::multiplies<::__half>>
{
  using type = cub::detail::transform::tile::tile_multiplies;
};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct tile_eligible<::cuda::std::plus<::__nv_bfloat16>, ::__nv_bfloat16, 2> : ::cuda::std::true_type
{};
template <>
struct tile_eligible<::cuda::std::multiplies<::__nv_bfloat16>, ::__nv_bfloat16, 2> : ::cuda::std::true_type
{};
template <>
struct tile_operator<::cuda::std::plus<::__nv_bfloat16>>
{
  using type = cub::detail::transform::tile::tile_plus;
};
template <>
struct tile_operator<::cuda::std::multiplies<::__nv_bfloat16>>
{
  using type = cub::detail::transform::tile::tile_multiplies;
};
#  endif // _CCCL_HAS_NVBF16()
} // namespace transform

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
