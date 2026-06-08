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
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_ARGUMENT

//! @brief Sentinel type indicating no bounds are present.
struct __no_bounds
{};

// =====================================================================
// static_bounds
// =====================================================================

//! @brief Compile-time bounds on an argument value.
//!
//! The bound type is deduced from the non-type template parameters.
//!
//! @tparam _Lower The static lower bound.
//! @tparam _Upper The static upper bound.
template <auto _Lower, auto _Upper>
struct __static_bounds
{
  static_assert(::cuda::std::is_same_v<decltype(_Lower), decltype(_Upper)>,
                "Static bounds endpoints must have the same type");
  static_assert(_Lower <= _Upper, "Lower bound must be <= upper bound");

  [[nodiscard]] _CCCL_API static constexpr decltype(_Lower) lower() noexcept
  {
    return _Lower;
  }
  [[nodiscard]] _CCCL_API static constexpr decltype(_Upper) upper() noexcept
  {
    return _Upper;
  }
};

template <class _Tp>
inline constexpr bool __is_static_bounds_v = false;
template <auto _Lower, auto _Upper>
inline constexpr bool __is_static_bounds_v<__static_bounds<_Lower, _Upper>> = true;

// =====================================================================
// runtime_bounds
// =====================================================================

//! @brief Runtime bounds on an argument value.
//!
//! @tparam _Tp The value type of the bounds.
template <class _Tp>
struct __runtime_bounds
{
  _Tp __lower_ = ::cuda::std::numeric_limits<_Tp>::lowest();
  _Tp __upper_ = (::cuda::std::numeric_limits<_Tp>::max)();

  constexpr __runtime_bounds() noexcept = default;

  _CCCL_API constexpr __runtime_bounds(_Tp __lower, _Tp __upper) noexcept
      : __lower_(__lower)
      , __upper_(__upper)
  {
    _CCCL_ASSERT(__lower <= __upper, "Runtime lower bound must be <= runtime upper bound");
  }

  [[nodiscard]] _CCCL_API constexpr _Tp lower() const noexcept
  {
    return __lower_;
  }

  [[nodiscard]] _CCCL_API constexpr _Tp upper() const noexcept
  {
    return __upper_;
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Tp>
_CCCL_HOST_DEVICE __runtime_bounds(_Tp, _Tp) -> __runtime_bounds<_Tp>;
#endif // _CCCL_DOXYGEN_INVOKED

template <class _Tp>
inline constexpr bool __is_runtime_bounds_v = false;
template <class _Tp>
inline constexpr bool __is_runtime_bounds_v<__runtime_bounds<_Tp>> = true;

// =====================================================================
// bounds — factory functions
// =====================================================================

//! @brief Create compile-time bounds.
//!
//! @tparam _Lower The static lower bound.
//! @tparam _Upper The static upper bound.
//! @return A compile-time bounds object.
template <auto _Lower, auto _Upper>
[[nodiscard]] _CCCL_API constexpr __static_bounds<_Lower, _Upper> __bounds() noexcept
{
  return {};
}

//! @brief Create runtime bounds.
//!
//! @param __lower The runtime lower bound.
//! @param __upper The runtime upper bound.
//! @return A runtime bounds object.
template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __runtime_bounds<_Tp> __bounds(_Tp __lower, _Tp __upper) noexcept
{
  return {__lower, __upper};
}

template <class _Tp>
inline constexpr bool __is_static_bounds_cv_v = __is_static_bounds_v<::cuda::std::remove_cvref_t<_Tp>>;
template <class _Tp>
inline constexpr bool __is_runtime_bounds_cv_v = __is_runtime_bounds_v<::cuda::std::remove_cvref_t<_Tp>>;
template <class _Tp>
inline constexpr bool __is_bounds_v = __is_static_bounds_cv_v<_Tp> || __is_runtime_bounds_cv_v<_Tp>;

// =====================================================================
// __assert_in_range
// =====================================================================

template <class _To, class _From>
_CCCL_API constexpr void __assert_in_range([[maybe_unused]] _From __val) noexcept
{
  if constexpr (::cuda::std::__cccl_is_integer_v<_To> && ::cuda::std::__cccl_is_integer_v<_From>)
  {
    _CCCL_ASSERT(::cuda::std::in_range<_To>(__val), "runtime bound value overflows the element type");
  }
}

template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To __runtime_bound_cast(_From __val) noexcept
{
  __assert_in_range<_To>(__val);
  return static_cast<_To>(__val);
}

template <class _To, auto _Value>
_CCCL_API constexpr bool __static_bound_in_range() noexcept
{
  using _From = decltype(_Value);

  if constexpr (::cuda::std::__cccl_is_integer_v<_To> && ::cuda::std::__cccl_is_integer_v<_From>)
  {
    return ::cuda::std::in_range<_To>(_Value);
  }
  else if constexpr (::cuda::std::is_arithmetic_v<_To> && ::cuda::std::is_arithmetic_v<_From>)
  {
    return static_cast<_From>(static_cast<_To>(_Value)) == _Value;
  }
  else
  {
    return true;
  }
}

template <class _ElementType, class _StaticBounds>
inline constexpr bool __valid_static_bounds_v = true;

template <class _ElementType, auto _Lowest, auto _Highest>
inline constexpr bool __valid_static_bounds_v<_ElementType, __static_bounds<_Lowest, _Highest>> =
  ::cuda::__argument::__static_bound_in_range<_ElementType, _Lowest>()
  && ::cuda::__argument::__static_bound_in_range<_ElementType, _Highest>();

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_lowest() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::lowest();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::lower());
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_highest() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return (::cuda::std::numeric_limits<_ElementType>::max)();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::upper());
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __effective_lowest(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  auto __static_lowest = ::cuda::__argument::__wrapper_static_lowest<_ElementType, _StaticBounds>();
  return __static_lowest > __runtime_bounds.lower() ? __static_lowest : __runtime_bounds.lower();
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __effective_highest(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  auto __static_highest = ::cuda::__argument::__wrapper_static_highest<_ElementType, _StaticBounds>();
  return __static_highest < __runtime_bounds.upper() ? __static_highest : __runtime_bounds.upper();
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr bool __valid_argument_bounds(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  return ::cuda::__argument::__effective_lowest<_ElementType, _StaticBounds>(__runtime_bounds)
      <= ::cuda::__argument::__effective_highest<_ElementType, _StaticBounds>(__runtime_bounds);
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr void __validate_bounds_intersection(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  static_assert(__valid_static_bounds_v<_ElementType, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");
  _CCCL_VERIFY((::cuda::__argument::__valid_argument_bounds<_ElementType, _StaticBounds>(__runtime_bounds)),
               "static and runtime argument bounds do not intersect");
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr void __validate_static_element_bounds([[maybe_unused]] const _ElementType& __val) noexcept
{
  if constexpr (!::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    _CCCL_ASSERT((__val >= ::cuda::__argument::__wrapper_static_lowest<_ElementType, _StaticBounds>()),
                 "immediate argument value is below static lowest bound");
    _CCCL_ASSERT((__val <= ::cuda::__argument::__wrapper_static_highest<_ElementType, _StaticBounds>()),
                 "immediate argument value is above static highest bound");
  }
}

template <class _ElementType>
_CCCL_API constexpr void __validate_runtime_element_bounds(
  [[maybe_unused]] const _ElementType& __val, [[maybe_unused]] __runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  _CCCL_ASSERT((__val >= __runtime_bounds.lower()), "immediate argument value is below runtime lower bound");
  _CCCL_ASSERT((__val <= __runtime_bounds.upper()), "immediate argument value is above runtime upper bound");
}

_CCCL_END_NAMESPACE_CUDA_ARGUMENT

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_BOUNDS_H
