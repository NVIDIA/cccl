// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-time policy for cub::DeviceTransform's tile path.
//
// INTERNAL EXTENSION POINTS (cub::detail::transform::tile) -- two independent axes:
//   tile_eligible_v<Op, T, NumInputs> -- specialize to true to opt a (functor type,
//                                   element type, input arity) combo into the
//                                   tile dispatch path. Eligibility only.
//   tile_operator<Op>           -- the __tile__ functor the tile kernel runs
//                                   for Op. No default: every tile-eligible Op
//                                   must specialize it with `using type = <a
//                                   stateless __tile__ functor mirroring Op>`,
//                                   because a scalar functor (e.g.
//                                   cuda::std::plus<__half>) cannot be invoked
//                                   on ct::tile. Omitting it is a clear
//                                   static_assert, not a cryptic kernel error.
//   tile_operator_t<Op>         -- alias for tile_operator<Op>::type.
//   tile_mufu_heavy_v<Op>       -- specialize to true to flag Op as MUFU-heavy; the tile policy picker uses it.
//
// Eligibility ("may this combo use the tile path?") and substitution ("which
// __tile__ functor do we actually run?") are separate traits, so an eligible op
// always registers both: tile_eligible_v<Op,T,NumInputs> and tile_operator<Op>.

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
#  include <cuda/std/__type_traits/is_extended_arithmetic.h>

#  include <cuda_tile.h>

CUB_NAMESPACE_BEGIN

// Internal extension surface.
namespace detail::transform::tile
{
// Opt a (functor type, element type, input arity) combo into the tile dispatch path: specialize this to
// true for the combo. Eligibility only -- the __tile__ functor to actually run is named by tile_operator<Op>.
// The kernel default-constructs tile_operator_t<Op> and never sees the Op instance, so the substitute must be
// stateless (the dispatch static_assert enforces this). Op may carry state; it is only used on the fallback path.
template <typename Op, typename T, ::cuda::std::size_t NumInputs>
inline constexpr bool tile_eligible_v = false;

// The __tile__ functor the tile kernel runs for Op.
template <typename Op>
struct tile_operator
{
  static_assert(sizeof(Op) == 0,
                "cub::detail::transform::tile::tile_operator<Op> must be specialized for every tile-eligible Op: "
                "provide `using type = <stateless __tile__ functor mirroring Op>`.");
};

template <typename Op>
using tile_operator_t = typename tile_operator<Op>::type;

// Hint that Op uses MUFU (multi-function unit, sin/cos/exp/log/tanh/rcp/rsq); specialize to true to make the tile
// policy picker cap items/thread so MUFU pipes are not oversaturated.
template <typename Op>
inline constexpr bool tile_mufu_heavy_v = false;
} // namespace detail::transform::tile

// Built-in trait specializations.
namespace detail::transform::tile
{
// cuda::std::plus/multiplies route to the tile kernel for every element type it supports, so the experimental
// opt-in exercises the tile path broadly. __int128 is excluded (no tensor_span/partition_view support).
// The transparent cuda::std::plus<>/multiplies<> have a templated operator() that is tile-callable, so they
// serve directly as the tile_operator for the typed cuda::std::plus<T>/multiplies<T> a user passes.
template <typename T>
inline constexpr bool tile_supported_element_v = ::cuda::std::__is_extended_arithmetic_v<T> && sizeof(T) <= 8;

template <typename T>
inline constexpr bool tile_eligible_v<::cuda::std::plus<T>, T, 2> = tile_supported_element_v<T>;
template <typename T>
inline constexpr bool tile_eligible_v<::cuda::std::plus<>, T, 2> = tile_supported_element_v<T>;
template <typename T>
inline constexpr bool tile_eligible_v<::cuda::std::multiplies<T>, T, 2> = tile_supported_element_v<T>;
template <typename T>
inline constexpr bool tile_eligible_v<::cuda::std::multiplies<>, T, 2> = tile_supported_element_v<T>;

template <typename T>
struct tile_operator<::cuda::std::plus<T>>
{
  using type = ::cuda::std::plus<>;
};
template <typename T>
struct tile_operator<::cuda::std::multiplies<T>>
{
  using type = ::cuda::std::multiplies<>;
};
} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CUB_HAS_TILE_TRANSFORM()
