//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_EXTENT_CUH
#define _CUDAX___CUCO___DETAIL_EXTENT_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__mdspan/extents.h>

#include <cuda/experimental/__cuco/__detail/prime.hpp>
#include <cuda/experimental/__cuco/__detail/types.cuh>
#include <cuda/experimental/__cuco/__detail/utils.hpp>
#include <cuda/experimental/__cuco/probing_scheme.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
/// @brief A valid (post-rounding) extent type.
template <class _SizeType>
using __valid_extent = extent<_SizeType, dynamic_extent>;

//! @brief Computes a valid storage extent for double hashing.
//!
//! @param __ext Requested storage extent
template <int _CgSize, int _BucketSize, class _SizeType, ::cuda::std::size_t _Extent>
[[nodiscard]] _CCCL_API constexpr auto __make_valid_extent_double_hash(extent<_SizeType, _Extent> __ext)
{
  constexpr auto __stride = _CgSize * _BucketSize;

  const auto __size =
    ::cuda::ceil_div(::cuda::std::max(static_cast<_SizeType>(__ext.extent(0)), static_cast<_SizeType>(1)), __stride);
  if (__size > ::cuda::std::numeric_limits<_SizeType>::max() / static_cast<_SizeType>(__stride))
  {
    _CCCL_THROW(::std::logic_error, "Invalid input extent");
  }

  return __valid_extent<_SizeType>{static_cast<_SizeType>(
    ::cuda::experimental::cuco::__detail::__next_prime(static_cast<::cuda::std::uint64_t>(__size)) * __stride)};
}

//! @brief Computes a valid storage extent for a given probing scheme.
//!
//! @tparam _ProbingScheme Probing scheme type
//! @tparam _BucketSize Bucket size
//! @tparam _SizeType Size type
//! @tparam _Extent Extent value
//!
//! @param __ext Requested storage extent
template <class _ProbingScheme, int _BucketSize, class _SizeType, ::cuda::std::size_t _Extent>
[[nodiscard]] _CCCL_API constexpr auto __make_valid_extent(extent<_SizeType, _Extent> __ext)
{
  if constexpr (is_double_hashing<_ProbingScheme>::value)
  {
    return __make_valid_extent_double_hash<_ProbingScheme::cg_size, _BucketSize, _SizeType, _Extent>(__ext);
  }
  else
  {
    constexpr auto __stride = _ProbingScheme::cg_size * _BucketSize;
    const auto __size =
      ::cuda::ceil_div(::cuda::std::max(static_cast<_SizeType>(__ext.extent(0)), static_cast<_SizeType>(1)), __stride)
      + static_cast<_SizeType>(__ext.extent(0) == 0);
    return __valid_extent<_SizeType>{__size * __stride};
  }
}

//! @brief Computes a valid storage extent with a desired load factor.
//!
//! @param __ext Requested storage extent
//! @param __desired_load_factor Desired load factor (0, 1]
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_API constexpr auto __make_valid_extent(extent<_SizeType> __ext, double __desired_load_factor)
{
  if (__desired_load_factor <= 0.)
  {
    _CCCL_THROW(::std::logic_error, "Desired occupancy must be larger than zero");
  }
  if (__desired_load_factor > 1.)
  {
    _CCCL_THROW(::std::logic_error, "Desired occupancy must be no larger than one");
  }

  const auto __temp = ::cuda::std::ceil(static_cast<double>(_SizeType{__ext.extent(0)}) / __desired_load_factor);
  if (__temp > static_cast<double>(::cuda::std::numeric_limits<_SizeType>::max()))
  {
    _CCCL_THROW(::std::logic_error,
                "Invalid load factor: requested extent divided by load factor exceeds maximum representable value");
  }
  return __make_valid_extent<_ProbingScheme, _BucketSize>(extent<_SizeType>{static_cast<_SizeType>(__temp)});
}

//! @brief Convenience overload taking a raw size.
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_API constexpr auto __make_valid_extent(_SizeType __size)
{
  return __make_valid_extent<_ProbingScheme, _BucketSize, _SizeType, dynamic_extent>(extent<_SizeType>{__size});
}

//! @brief Compile-time adjusted total slot count for a static extent _N.
//!
//! Applies the same prime/stride logic as `__make_valid_extent` but produces
//! the result as a compile-time constant suitable for use in non-type template
//! arguments (e.g., `cuda::std::extents<size_t, __valid_extent_v<PS, BS, N>>`).
//!
//! @note Requires _N != dynamic_extent.
//! @note Idempotent: applying twice with the same probing scheme and bucket size
//! returns the same value, so it is safe to pass an already-adjusted N.
template <class _ProbingScheme, int _BucketSize, ::cuda::std::size_t _N>
_CCCL_HIDE_FROM_ABI constexpr ::cuda::std::size_t __valid_extent_v = static_cast<::cuda::std::size_t>(
  __make_valid_extent<_ProbingScheme, _BucketSize>(extent<::cuda::std::size_t, _N>{}).extent(0));

//! @brief Compile-time adjusted slot count, with `dynamic_extent` passthrough.
//!
//! Companion to `__valid_extent_v` that accepts `cuda::std::dynamic_extent`:
//! - Static `_Capacity`  → prime/stride-adjusted value (same as `__valid_extent_v`).
//! - `dynamic_extent`    → `dynamic_extent` (actual size known only at runtime).
//!
//! Shared by all open-addressing containers (`static_map`, `static_map_ref`,
//! and any future `static_set`/`static_multimap`/...) so the requested→actual
//! mapping is computed in one place.
template <class _ProbingScheme, int _BucketSize, ::cuda::std::size_t _Capacity>
_CCCL_HIDE_FROM_ABI constexpr ::cuda::std::size_t __valid_capacity_v =
  (_Capacity == ::cuda::std::dynamic_extent)
    ? ::cuda::std::dynamic_extent
    : __valid_extent_v<_ProbingScheme, _BucketSize, _Capacity>;

//! @brief Runtime adjusted slot count for an open-addressing container.
//!
//! Convenience wrapper over `__make_valid_extent` returning the adjusted size
//! as a raw `_SizeType` rather than an `extent`.
template <class _ProbingScheme, int _BucketSize, class _SizeType>
[[nodiscard]] _CCCL_API constexpr _SizeType __valid_capacity(_SizeType __requested)
{
  return static_cast<_SizeType>(
    __make_valid_extent<_ProbingScheme, _BucketSize>(extent<_SizeType>{__requested}).extent(0));
}
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_EXTENT_CUH
