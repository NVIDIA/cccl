//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_ISCLOSE_H
#define _CUDA___NUMERIC_ISCLOSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/__complex/get_real_imag.h>
#include <cuda/__complex/traits.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/__utility/in_range.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
using __isclose_compare_t _CCCL_NODEBUG_ALIAS = ::cuda::std::
  conditional_t<(::cuda::std::__is_extended_floating_point_v<_Tp> && sizeof(_Tp) <= sizeof(float)), float, _Tp>;

// compute 10^-(digits10 / 2)
template <typename _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL float __isclose_default_relative_tolerance() noexcept
{
  constexpr auto __digits = ::cuda::ceil_div(::cuda::std::numeric_limits<_Tp>::max_digits10, 2);
  auto __exp              = 1.0f;
  for (int __i = 0; __i < __digits; ++__i)
  {
    __exp *= 10.0f;
  }
  return 1.0f / __exp;
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr bool
__isclose_fp_impl(const _Tp __lhs, const _Tp __rhs, const float __rel_tol, const _Tp __abs_tol) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__rel_tol, 0.0f, 1.0f),
               "cuda::isclose: relative tolerance must be in the range [0.0, 1.0]");
  _CCCL_ASSERT(::cuda::std::isfinite(__abs_tol) && __abs_tol >= _Tp{0},
               "cuda::isclose: absolute tolerance must be finite and non-negative");
  if (__lhs == __rhs)
  {
    return true;
  }
  if (!::cuda::std::isfinite(__lhs) || !::cuda::std::isfinite(__rhs))
  {
    return false;
  }
  const auto __diff      = ::cuda::std::fabs(__lhs - __rhs);
  const auto __lhs_abs   = ::cuda::std::fabs(__lhs);
  const auto __rhs_abs   = ::cuda::std::fabs(__rhs);
  const auto __rel_value = static_cast<_Tp>(__rel_tol * ::cuda::std::fmax(__lhs_abs, __rhs_abs));
  return __diff <= ::cuda::std::fmax(__abs_tol, __rel_value);
}

template <typename _ComplexType, typename _AbsTol>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __isclose_complex_impl(
  const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol, const _AbsTol __abs_tol) noexcept
{
  using __scalar_t _CCCL_NODEBUG_ALIAS  = typename _ComplexType::value_type;
  using __compare_t _CCCL_NODEBUG_ALIAS = __isclose_compare_t<__scalar_t>;
#if _CCCL_HAS_FLOAT128()
  // __float128 is not supported because cuda::std::hypot is not implemented for this type
  static_assert(!::cuda::std::is_same_v<__scalar_t, __float128>, "cuda::isclose: __float128 is not supported");
#endif // _CCCL_HAS_FLOAT128()
  _CCCL_ASSERT(::cuda::in_range(__rel_tol, 0.0f, 1.0f),
               "cuda::isclose: relative tolerance must be in the range [0.0, 1.0]");
  _CCCL_ASSERT(::cuda::std::isfinite(__abs_tol) && __abs_tol >= __scalar_t{0},
               "cuda::isclose: absolute tolerance must be finite and non-negative");

  const auto __lhs_real = static_cast<__compare_t>(::cuda::__get_real(__lhs));
  const auto __lhs_imag = static_cast<__compare_t>(::cuda::__get_imag(__lhs));
  const auto __rhs_real = static_cast<__compare_t>(::cuda::__get_real(__rhs));
  const auto __rhs_imag = static_cast<__compare_t>(::cuda::__get_imag(__rhs));
  const auto __abs      = static_cast<__compare_t>(__abs_tol);

  if (__lhs_real == __rhs_real && __lhs_imag == __rhs_imag)
  {
    return true;
  }
  if (!::cuda::std::isfinite(__lhs_real) || !::cuda::std::isfinite(__lhs_imag) || !::cuda::std::isfinite(__rhs_real)
      || !::cuda::std::isfinite(__rhs_imag))
  {
    return false;
  }
  const auto __diff      = ::cuda::std::hypot(__lhs_real - __rhs_real, __lhs_imag - __rhs_imag);
  const auto __lhs_abs   = ::cuda::std::hypot(__lhs_real, __lhs_imag);
  const auto __rhs_abs   = ::cuda::std::hypot(__rhs_real, __rhs_imag);
  const auto __rel_value = __rel_tol * ::cuda::std::fmax(__lhs_abs, __rhs_abs);
  return __diff <= ::cuda::std::fmax(__abs, __rel_value);
}

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr ::cuda::std::make_unsigned_t<_Tp>
__safe_abs_diff(const _Tp __lhs, const _Tp __rhs) noexcept
{
  using __unsigned_t _CCCL_NODEBUG_ALIAS = ::cuda::std::make_unsigned_t<_Tp>;
  const auto __lhs_abs                   = ::cuda::uabs(__lhs);
  const auto __rhs_abs                   = ::cuda::uabs(__rhs);
  const auto __is_lhs_negative           = ::cuda::std::cmp_less(__lhs, _Tp{0});
  const auto __is_rhs_negative           = ::cuda::std::cmp_less(__rhs, _Tp{0});
  if (__is_lhs_negative != __is_rhs_negative)
  {
    return static_cast<__unsigned_t>(__lhs_abs + __rhs_abs);
  }
  return (__lhs_abs < __rhs_abs)
         ? static_cast<__unsigned_t>(__rhs_abs - __lhs_abs)
         : static_cast<__unsigned_t>(__lhs_abs - __rhs_abs);
}

_CCCL_TEMPLATE(typename _Unsigned, typename _Float)
_CCCL_REQUIRES(::cuda::std::is_unsigned_v<_Unsigned>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __is_floating_point_convertible_to_unsigned(const _Float __x) noexcept
{
  constexpr int __unsigned_bits = ::cuda::std::numeric_limits<_Unsigned>::digits;
  if constexpr (__unsigned_bits >= ::cuda::std::numeric_limits<_Float>::max_exponent)
  {
    return true;
  }
  else
  {
    return __x < ::cuda::std::ldexp(_Float{1}, __unsigned_bits);
  }
}

// Edge cases for 128-bit integers can be not representable as 32-bit float. In addition, large 128-bit values
// rounding to floatint-point values can cause precision loss.
template <typename _Tp>
using __isclose_integer_rel_t _CCCL_NODEBUG_ALIAS = ::cuda::std::conditional_t<
#if _CCCL_HAS_INT128()
  sizeof(_Tp) == sizeof(__int128_t),
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  false,
#endif // !_CCCL_HAS_INT128()
  double,
  float>;

[[nodiscard]] _CCCL_HOST_DEVICE_API int __extract_power_of_two_shift(const float __rel_tol) noexcept
{
  int __exponent                 = 0;
  const auto __normalized_mant   = ::cuda::std::frexp(__rel_tol, &__exponent);
  constexpr auto __pow2_mantissa = 0.5f;
  if (__normalized_mant != __pow2_mantissa) // not a power of two
  {
    return -1;
  }
  return 1 - __exponent;
}

template <typename _Tp>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool
__isclose_integer_impl(const _Tp __lhs, const _Tp __rhs, const float __rel_tol, const _Tp __abs_tol) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__rel_tol, 0.0f, 1.0f),
               "cuda::isclose: relative tolerance must be in the range [0.0, 1.0]");
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__abs_tol, _Tp{0}),
               "cuda::isclose: absolute tolerance must be non-negative");
  using __unsigned_t _CCCL_NODEBUG_ALIAS = ::cuda::std::make_unsigned_t<_Tp>;
  const auto __lhs_abs                   = ::cuda::uabs(__lhs);
  const auto __rhs_abs                   = ::cuda::uabs(__rhs);
  const auto __diff                      = ::cuda::__safe_abs_diff(__lhs, __rhs);
  const auto __abs                       = static_cast<__unsigned_t>(__abs_tol);
  const auto __max_abs                   = ::cuda::std::max(__lhs_abs, __rhs_abs);
  // if the relative tolerance is exactly a power of two, we can use bit shifts to compute the relative value without
  // potentially losing precision
  const auto __pow2_rel_tol_shift = ::cuda::__extract_power_of_two_shift(__rel_tol);
  if (__pow2_rel_tol_shift >= 0)
  {
    constexpr int __digits = ::cuda::std::numeric_limits<__unsigned_t>::digits;
    const auto __rel_value = (__pow2_rel_tol_shift >= __digits) ? __unsigned_t{0} : (__max_abs >> __pow2_rel_tol_shift);
    return __diff <= ::cuda::std::max(__abs, __rel_value);
  }
  using __rel_value_t _CCCL_NODEBUG_ALIAS = ::cuda::__isclose_integer_rel_t<_Tp>;
  const auto __rel_value = static_cast<__rel_value_t>(__rel_tol) * static_cast<__rel_value_t>(__max_abs);
  // if the floating-point value is too large to be convertible to an unsigned type,
  // then __diff is always less than __rel_value
  if (!::cuda::__is_floating_point_convertible_to_unsigned<__unsigned_t>(__rel_value))
  {
    return true;
  }
  return __diff <= ::cuda::std::max(__abs, static_cast<__unsigned_t>(__rel_value));
}

//----------------------------------------------------------------------------------------------------------------------
// Public API

// Scalar overloads

//! @brief Checks whether two arithmetic values are close to each other using a relative and absolute tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @param __rel_tol The relative tolerance.
//! @param __abs_tol The absolute tolerance.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> || ::cuda::is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool
isclose(const _Tp __lhs, const _Tp __rhs, const float __rel_tol, const _Tp __abs_tol) noexcept
{
  if constexpr (::cuda::std::__cccl_is_integer_v<_Tp>)
  {
    return ::cuda::__isclose_integer_impl(+__lhs, +__rhs, __rel_tol, +__abs_tol);
  }
  else
  {
    using __value_t _CCCL_NODEBUG_ALIAS = __isclose_compare_t<_Tp>;
    return ::cuda::__isclose_fp_impl(
      static_cast<__value_t>(__lhs), static_cast<__value_t>(__rhs), __rel_tol, static_cast<__value_t>(__abs_tol));
  }
}

//! @brief Checks whether two arithmetic values are close to each other using a relative tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @param __rel_tol The relative tolerance.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> || ::cuda::is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool isclose(const _Tp __lhs, const _Tp __rhs, const float __rel_tol) noexcept
{
  return ::cuda::isclose(__lhs, __rhs, __rel_tol, _Tp{0});
}

//! @brief Checks whether two arithmetic values are close to each other using the default relative tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> || ::cuda::is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool isclose(const _Tp __lhs, const _Tp __rhs) noexcept
{
  if constexpr (::cuda::std::__cccl_is_integer_v<_Tp>)
  {
    return __lhs == __rhs;
  }
  else
  {
    constexpr auto __rel_tol = ::cuda::__isclose_default_relative_tolerance<_Tp>();
    return ::cuda::isclose(__lhs, __rhs, __rel_tol, _Tp{0});
  }
}

// Complex overloads

template <typename _Tp, typename _AbsTol, bool = __is_any_complex_v<_Tp>>
inline constexpr bool __isclose_complex_comparison_v = false;

template <typename _Tp, typename _AbsTol>
inline constexpr bool __isclose_complex_comparison_v<_Tp, _AbsTol, true> =
  ::cuda::std::is_same_v<typename _Tp::value_type, _AbsTol>;

//! @brief Checks whether two complex values are close to each other using a relative and absolute tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @param __rel_tol The relative tolerance.
//! @param __abs_tol The absolute tolerance.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _ComplexType, typename _AbsTol)
_CCCL_REQUIRES(__isclose_complex_comparison_v<_ComplexType, _AbsTol>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool
isclose(const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol, const _AbsTol __abs_tol) noexcept
{
  return ::cuda::__isclose_complex_impl(__lhs, __rhs, __rel_tol, __abs_tol);
}

//! @brief Checks whether two complex values are close to each other using a relative tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @param __rel_tol The relative tolerance.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _ComplexType)
_CCCL_REQUIRES(__is_any_complex_v<_ComplexType>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool
isclose(const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol) noexcept
{
  using __scalar_t _CCCL_NODEBUG_ALIAS = typename _ComplexType::value_type;
  return ::cuda::isclose(__lhs, __rhs, __rel_tol, __scalar_t{0});
}

//! @brief Checks whether two complex values are close to each other using the default relative tolerance.
//!
//! @param __lhs The first value to compare.
//! @param __rhs The second value to compare.
//! @return True if __lhs and __rhs are close to each other, false otherwise.
_CCCL_TEMPLATE(typename _ComplexType)
_CCCL_REQUIRES(__is_any_complex_v<_ComplexType>)
[[nodiscard]] _CCCL_HOST_DEVICE_API bool isclose(const _ComplexType& __lhs, const _ComplexType& __rhs) noexcept
{
  using __scalar_t _CCCL_NODEBUG_ALIAS = typename _ComplexType::value_type;
  return ::cuda::isclose(__lhs, __rhs, ::cuda::__isclose_default_relative_tolerance<__scalar_t>(), __scalar_t{0});
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_ISCLOSE_H
