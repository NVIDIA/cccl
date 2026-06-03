// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compile-time policy for cub::DeviceTransform's tile path.
//
// tile_eligible_v<Op, T, NIn> answers "should DeviceTransform::Transform
// route to the tile kernel for this (functor, element type, input arity)?".
// tile_mufu_heavy_v<Op> hints the tile policy picker that Op spends most of
// its time on MUFU instructions, so the picker caps items/thread at the
// vector width to avoid piling up MUFU work that cannot SIMD on Blackwell
// for sub-4-byte types.
//
// This header is pure trait infrastructure; no callers yet. Specializations
// land here as benches confirm tile wins for a (Op, T, NIn) combination.

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

CUB_NAMESPACE_BEGIN

namespace detail::transform::tile
{

// Primary template: tile path is opt-in. Specialize for combinations where a
// bench has shown the tile kernel beats the existing CUB algorithms.
template <typename Op, typename T, ::cuda::std::size_t NIn>
struct tile_eligible : ::cuda::std::false_type
{};

template <typename Op, typename T, ::cuda::std::size_t NIn>
inline constexpr bool tile_eligible_v = tile_eligible<Op, T, NIn>::value;

// Companion trait: report Op as MUFU-heavy so the tile policy picker caps
// items/thread at the vector width on small element types. Default is false.
template <typename Op>
struct tile_mufu_heavy : ::cuda::std::false_type
{};

template <typename Op>
inline constexpr bool tile_mufu_heavy_v = tile_mufu_heavy<Op>::value;

#  if _CCCL_HAS_NVFP16()
template <>
struct tile_eligible<::cuda::std::plus<__half>, __half, 2> : ::cuda::std::true_type
{};
template <>
struct tile_eligible<::cuda::std::multiplies<__half>, __half, 2> : ::cuda::std::true_type
{};
#  endif // _CCCL_HAS_NVFP16()

#  if _CCCL_HAS_NVBF16()
template <>
struct tile_eligible<::cuda::std::plus<__nv_bfloat16>, __nv_bfloat16, 2> : ::cuda::std::true_type
{};
template <>
struct tile_eligible<::cuda::std::multiplies<__nv_bfloat16>, __nv_bfloat16, 2> : ::cuda::std::true_type
{};
#  endif // _CCCL_HAS_NVBF16()

} // namespace detail::transform::tile

CUB_NAMESPACE_END

#endif // _CCCL_CTK_AT_LEAST(13, 3)
