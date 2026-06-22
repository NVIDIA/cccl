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
#include <cuda/__complex/get_real_imag.h>
#include <cuda/__complex/traits.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cmath/abs.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/promote.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
using __isclose_comparison_t = ::cuda::std::__promote_t<_Tp>;

template <class _Tp, class _Up>
using __isclose_comparison2_t = ::cuda::std::__promote_t<__isclose_comparison_t<_Tp>, _Up>;

template <class _Tp, bool = ::cuda::std::__promote<_Tp>::value>
inline constexpr bool __isclose_has_comparison_v = false;

template <class _Tp>
inline constexpr bool __isclose_has_comparison_v<_Tp, true> = true;

template <class _Tp, class _Up, bool = __isclose_has_comparison_v<_Tp>>
inline constexpr bool __isclose_has_comparison2_v = false;

template <class _Tp, class _Up>
inline constexpr bool __isclose_has_comparison2_v<_Tp, _Up, true> =
  ::cuda::std::__promote<__isclose_comparison_t<_Tp>, _Up>::value;

template <class _Tp, class _AbsTol, bool = __isclose_has_comparison2_v<_Tp, _AbsTol>>
inline constexpr bool __isclose_has_abs_tol_v = false;

template <class _Tp, class _AbsTol>
inline constexpr bool __isclose_has_abs_tol_v<_Tp, _AbsTol, true> =
  ::cuda::std::is_same_v<__isclose_comparison_t<_Tp>, __isclose_comparison2_t<_Tp, _AbsTol>>;

template <class _ComplexType, bool = __is_cccl_complex_v<_ComplexType>>
inline constexpr bool __isclose_has_complex_comparison_v = false;

template <class _ComplexType>
inline constexpr bool __isclose_has_complex_comparison_v<_ComplexType, true> =
  __isclose_has_comparison_v<typename _ComplexType::value_type>;

template <class _ComplexType, class _AbsTol, bool = __is_cccl_complex_v<_ComplexType>>
inline constexpr bool __isclose_has_complex_abs_tol_v = false;

template <class _ComplexType, class _AbsTol>
inline constexpr bool __isclose_has_complex_abs_tol_v<_ComplexType, _AbsTol, true> =
  __isclose_has_abs_tol_v<typename _ComplexType::value_type, _AbsTol>;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr float __isclose_default_rel_tol() noexcept
{
  constexpr auto __digits = ::cuda::ceil_div(::cuda::std::numeric_limits<_Tp>::max_digits10, 2);
  auto __tol              = 1.0f;
  for (int __i = 0; __i < __digits; ++__i)
  {
    __tol /= 10.0f;
  }
  return __tol;
}

template <class _Tp>
_CCCL_API constexpr void __isclose_validate_tolerances(const float __rel_tol, const _Tp __abs_tol) noexcept
{
  _CCCL_ASSERT(::cuda::std::isfinite(__rel_tol) && __rel_tol >= 0.0f,
               "cuda::isclose: relative tolerance must be finite and non-negative");
  _CCCL_ASSERT(::cuda::std::isfinite(__abs_tol) && __abs_tol >= _Tp{0},
               "cuda::isclose: absolute tolerance must be finite and non-negative");
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool __isclose_compare(
  const _Tp __diff, const _Tp __lhs_abs, const _Tp __rhs_abs, const float __rel_tol, const _Tp __abs_tol) noexcept
{
  return __diff <= ::cuda::std::max(__abs_tol, static_cast<_Tp>(__rel_tol) * ::cuda::std::max(__lhs_abs, __rhs_abs));
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr bool
__isclose_impl(const _Tp __lhs, const _Tp __rhs, const float __rel_tol, const _Tp __abs_tol) noexcept
{
  ::cuda::__isclose_validate_tolerances(__rel_tol, __abs_tol);

  if (__lhs == __rhs)
  {
    return true;
  }
  if (::cuda::std::isnan(__lhs) || ::cuda::std::isnan(__rhs))
  {
    return false;
  }
  if (::cuda::std::isinf(__lhs) || ::cuda::std::isinf(__rhs))
  {
    return false;
  }

  return ::cuda::__isclose_compare(
    ::cuda::std::abs(__lhs - __rhs), ::cuda::std::abs(__lhs), ::cuda::std::abs(__rhs), __rel_tol, __abs_tol);
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp __isclose_hypot(const _Tp __real_part, const _Tp __imag_part) noexcept
{
  return ::cuda::std::hypot(__real_part, __imag_part);
}

template <class _ComplexType, class _AbsTol>
[[nodiscard]] _CCCL_API bool __isclose_complex_impl(
  const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol, const _AbsTol __abs_tol) noexcept
{
  using _Value      = typename _ComplexType::value_type;
  using _Comparison = __isclose_comparison_t<_Value>;

  const auto __lhs_real = static_cast<_Comparison>(::cuda::__get_real(__lhs));
  const auto __lhs_imag = static_cast<_Comparison>(::cuda::__get_imag(__lhs));
  const auto __rhs_real = static_cast<_Comparison>(::cuda::__get_real(__rhs));
  const auto __rhs_imag = static_cast<_Comparison>(::cuda::__get_imag(__rhs));
  const auto __abs      = static_cast<_Comparison>(__abs_tol);

  ::cuda::__isclose_validate_tolerances(__rel_tol, __abs);

  if (__lhs_real == __rhs_real && __lhs_imag == __rhs_imag)
  {
    return true;
  }
  if (::cuda::std::isnan(__lhs_real) || ::cuda::std::isnan(__lhs_imag) || ::cuda::std::isnan(__rhs_real)
      || ::cuda::std::isnan(__rhs_imag))
  {
    return false;
  }
  if (::cuda::std::isinf(__lhs_real) || ::cuda::std::isinf(__lhs_imag) || ::cuda::std::isinf(__rhs_real)
      || ::cuda::std::isinf(__rhs_imag))
  {
    return false;
  }

  const auto __diff = ::cuda::__isclose_hypot(
    static_cast<_Comparison>(__lhs_real - __rhs_real), static_cast<_Comparison>(__lhs_imag - __rhs_imag));
  const auto __lhs_abs = ::cuda::__isclose_hypot(__lhs_real, __lhs_imag);
  const auto __rhs_abs = ::cuda::__isclose_hypot(__rhs_real, __rhs_imag);
  return ::cuda::__isclose_compare(__diff, __lhs_abs, __rhs_abs, __rel_tol, __abs);
}

//! @brief Checks whether two arithmetic values are close to each other using a relative and absolute tolerance.
_CCCL_TEMPLATE(class _Tp, class _AbsTol)
_CCCL_REQUIRES(__isclose_has_abs_tol_v<_Tp, _AbsTol>)
[[nodiscard]] _CCCL_API constexpr bool
isclose(const _Tp __lhs, const _Tp __rhs, const float __rel_tol, const _AbsTol __abs_tol) noexcept
{
  using _Comparison = __isclose_comparison_t<_Tp>;
  return ::cuda::__isclose_impl(
    static_cast<_Comparison>(__lhs), static_cast<_Comparison>(__rhs), __rel_tol, static_cast<_Comparison>(__abs_tol));
}

//! @brief Checks whether two arithmetic values are close to each other using a relative tolerance.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__isclose_has_comparison_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isclose(const _Tp __lhs, const _Tp __rhs, const float __rel_tol) noexcept
{
  using _Comparison = __isclose_comparison_t<_Tp>;
  return ::cuda::isclose(__lhs, __rhs, __rel_tol, _Comparison{0});
}

//! @brief Checks whether two arithmetic values are close to each other using the default relative tolerance.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__isclose_has_comparison_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isclose(const _Tp __lhs, const _Tp __rhs) noexcept
{
  using _Comparison = __isclose_comparison_t<_Tp>;
  return ::cuda::isclose(__lhs, __rhs, ::cuda::__isclose_default_rel_tol<_Comparison>(), _Comparison{0});
}

//! @brief Checks whether two complex values are close to each other using a relative and absolute tolerance.
_CCCL_TEMPLATE(class _ComplexType, class _AbsTol)
_CCCL_REQUIRES(__isclose_has_complex_abs_tol_v<_ComplexType, _AbsTol>)
[[nodiscard]] _CCCL_API bool
isclose(const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol, const _AbsTol __abs_tol) noexcept
{
  return ::cuda::__isclose_complex_impl(__lhs, __rhs, __rel_tol, __abs_tol);
}

//! @brief Checks whether two complex values are close to each other using a relative tolerance.
_CCCL_TEMPLATE(class _ComplexType)
_CCCL_REQUIRES(__isclose_has_complex_comparison_v<_ComplexType>)
[[nodiscard]] _CCCL_API bool
isclose(const _ComplexType& __lhs, const _ComplexType& __rhs, const float __rel_tol) noexcept
{
  using _Comparison = __isclose_comparison_t<typename _ComplexType::value_type>;
  return ::cuda::isclose(__lhs, __rhs, __rel_tol, _Comparison{0});
}

//! @brief Checks whether two complex values are close to each other using the default relative tolerance.
_CCCL_TEMPLATE(class _ComplexType)
_CCCL_REQUIRES(__isclose_has_complex_comparison_v<_ComplexType>)
[[nodiscard]] _CCCL_API bool isclose(const _ComplexType& __lhs, const _ComplexType& __rhs) noexcept
{
  using _Comparison = __isclose_comparison_t<typename _ComplexType::value_type>;
  return ::cuda::isclose(__lhs, __rhs, ::cuda::__isclose_default_rel_tol<_Comparison>(), _Comparison{0});
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_ISCLOSE_H
