//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ARGUMENT_ARGUMENT_BOUNDS_H
#define _CUDA___ARGUMENT_ARGUMENT_BOUNDS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Sentinel type indicating no bounds are present
struct __no_bounds
{};

// =====================================================================
// static_argument_bounds
// =====================================================================

//! @brief Compile-time bounds on an argument value.
//!
//! The value type is deduced from the non-type template parameters.
//!
//! @tparam _Lowest The static lower bound.
//! @tparam _Max The static upper bound.
template <auto _Lowest, auto _Max>
struct static_argument_bounds
{
  static_assert(::cuda::std::is_same_v<decltype(_Lowest), decltype(_Max)>, "Min and Max must have the same type");
  static_assert(_Lowest <= _Max, "Min must be <= Max");

  using value_type = decltype(_Lowest);

  static constexpr value_type lowest = _Lowest;
  static constexpr value_type max    = _Max;
};

// Helper to detect static_argument_bounds
template <class _Tp>
inline constexpr bool __is_static_argument_bounds_v = false;
template <auto _Lowest, auto _Max>
inline constexpr bool __is_static_argument_bounds_v<static_argument_bounds<_Lowest, _Max>> = true;

// =====================================================================
// runtime_argument_bounds
// =====================================================================

//! @brief Runtime bounds on an argument value.
//!
//! @tparam _Tp The value type of the bounds.
template <class _Tp>
struct runtime_argument_bounds
{
  using value_type = _Tp;

  _Tp lowest = ::cuda::std::numeric_limits<_Tp>::lowest();
  _Tp max    = ::cuda::std::numeric_limits<_Tp>::max();

  constexpr runtime_argument_bounds() noexcept = default;

  _CCCL_API constexpr runtime_argument_bounds(_Tp __lowest, _Tp __max) noexcept
      : lowest(__lowest)
      , max(__max)
  {
    _CCCL_ASSERT(__lowest <= __max, "Runtime lowest bound must be <= runtime max bound");
  }

  static constexpr bool has_runtime_bounds = true; // always true when explicitly constructed; see __no_bounds for
                                                   // "absent"
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Tp>
_CCCL_HOST_DEVICE runtime_argument_bounds(_Tp, _Tp) -> runtime_argument_bounds<_Tp>;
#endif // _CCCL_DOXYGEN_INVOKED

// Helper to detect runtime_argument_bounds
template <class _Tp>
inline constexpr bool __is_runtime_argument_bounds_v = false;
template <class _Tp>
inline constexpr bool __is_runtime_argument_bounds_v<runtime_argument_bounds<_Tp>> = true;

// =====================================================================
// argument_bounds — factory functions
// =====================================================================

//! @brief Create compile-time bounds.
//!
//! @code
//! cuda::argument_bounds<1, 8>()
//! @endcode
template <auto _Lowest, auto _Max>
[[nodiscard]] _CCCL_API constexpr static_argument_bounds<_Lowest, _Max> argument_bounds() noexcept
{
  return {};
}

//! @brief Create runtime bounds.
//!
//! @code
//! cuda::argument_bounds(rt_min, rt_max)
//! @endcode
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr runtime_argument_bounds<_Tp> argument_bounds(_Tp __min, _Tp __max) noexcept
{
  return {__min, __max};
}

// Helper to detect either bounds type (cv-qualified safe)
template <class _Tp>
inline constexpr bool __is_static_argument_bounds_cv_v = __is_static_argument_bounds_v<::cuda::std::remove_cv_t<_Tp>>;
template <class _Tp>
inline constexpr bool __is_runtime_argument_bounds_cv_v = __is_runtime_argument_bounds_v<::cuda::std::remove_cv_t<_Tp>>;
template <class _Tp>
inline constexpr bool __is_argument_bounds_v =
  __is_static_argument_bounds_cv_v<_Tp> || __is_runtime_argument_bounds_cv_v<_Tp>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_BOUNDS_H
