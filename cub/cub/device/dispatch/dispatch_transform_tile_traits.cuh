// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-time policy for cub::DeviceTransform's tile path.
//
// Users call cub::DeviceTransform::Transform with whatever scalar functor they
// have (e.g. cuda::std::plus<__half>). That functor is NOT directly callable
// from a tile transform_kernel -- its operator() takes scalars, not ct::tile.
// So eligible specializations declare a `tile_op_type` member that names a
// tile-friendly replacement functor (with __tile__ templated operator()) that
// performs the same operation. The dispatch hook then launches the tile
// kernel with the replacement, not the user's original.
//
// tile_eligible_v<Op, T, NIn> answers "should DeviceTransform::Transform
// route to the tile kernel for this (functor, element type, input arity)?".
// tile_mufu_heavy_v<Op> hints the tile policy picker that Op spends most of
// its time on MUFU instructions, so the picker caps items/thread at the
// vector width to avoid piling up MUFU work that cannot SIMD on Blackwell
// for sub-4-byte types.

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

#  include <cuda/std/__cccl/extended_data_types.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cstddef>

#  if _CCCL_TILE_COMPILATION()
#    include <cuda_tile.h>
#  endif

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

#  if _CCCL_TILE_COMPILATION()
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
#  endif // _CCCL_TILE_COMPILATION()

template <typename Op, typename T, ::cuda::std::size_t NIn>
struct tile_eligible : ::cuda::std::false_type
{};

template <typename Op, typename T, ::cuda::std::size_t NIn>
inline constexpr bool tile_eligible_v = tile_eligible<Op, T, NIn>::value;

template <typename Op>
struct tile_mufu_heavy : ::cuda::std::false_type
{};

template <typename Op>
inline constexpr bool tile_mufu_heavy_v = tile_mufu_heavy<Op>::value;

#  if _CCCL_TILE_COMPILATION()
#    if _CCCL_HAS_NVFP16()
template <>
struct tile_eligible<::cuda::std::plus<__half>, __half, 2> : ::cuda::std::true_type
{
  using tile_op_type = tile_plus;
};
template <>
struct tile_eligible<::cuda::std::multiplies<__half>, __half, 2> : ::cuda::std::true_type
{
  using tile_op_type = tile_multiplies;
};
#    endif // _CCCL_HAS_NVFP16()

#    if _CCCL_HAS_NVBF16()
template <>
struct tile_eligible<::cuda::std::plus<__nv_bfloat16>, __nv_bfloat16, 2> : ::cuda::std::true_type
{
  using tile_op_type = tile_plus;
};
template <>
struct tile_eligible<::cuda::std::multiplies<__nv_bfloat16>, __nv_bfloat16, 2> : ::cuda::std::true_type
{
  using tile_op_type = tile_multiplies;
};
#    endif // _CCCL_HAS_NVBF16()
#  endif // _CCCL_TILE_COMPILATION()

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CTK_AT_LEAST(13, 3)
