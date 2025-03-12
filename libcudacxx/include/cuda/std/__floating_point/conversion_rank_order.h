//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H
#define _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __fp_conv_rank_order
{
  __unordered,
  __greater,
  __equal,
  __less,
};

template <class _Lhs, class _Rhs>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp_conv_rank_order __fp_conv_rank_order_v_impl() noexcept
{
  static_assert(numeric_limits<_Lhs>::is_specialized, "numeric_limits<_Lhs> is not specialized");
  static_assert(numeric_limits<_Rhs>::is_specialized, "numeric_limits<_Rhs> is not specialized");

  if constexpr (numeric_limits<_Lhs>::min_exponent == numeric_limits<_Rhs>::min_exponent
                && numeric_limits<_Lhs>::max_exponent == numeric_limits<_Rhs>::max_exponent
                && numeric_limits<_Lhs>::digits == numeric_limits<_Rhs>::digits)
  {
#if _CCCL_HAS_LONG_DOUBLE()
    // If double and long double have the same properties, long double has the higher subrank
    if constexpr (numeric_limits<double>::min_exponent == numeric_limits<long double>::min_exponent
                  && numeric_limits<double>::max_exponent == numeric_limits<long double>::max_exponent
                  && numeric_limits<double>::digits == numeric_limits<long double>::digits)
    {
      if constexpr (_CCCL_TRAIT(is_same, _Lhs, long double) && !_CCCL_TRAIT(is_same, _Rhs, long double))
      {
        return __fp_conv_rank_order::__greater;
      }
      else if constexpr (!_CCCL_TRAIT(is_same, _Lhs, long double) && _CCCL_TRAIT(is_same, _Rhs, long double))
      {
        return __fp_conv_rank_order::__less;
      }
      else
      {
        return __fp_conv_rank_order::__equal;
      }
    }
    else
#endif // _CCCL_HAS_LONG_DOUBLE()
    {
      return __fp_conv_rank_order::__equal;
    }
  }
  else if constexpr (numeric_limits<_Lhs>::min_exponent <= numeric_limits<_Rhs>::min_exponent
                     && numeric_limits<_Lhs>::max_exponent >= numeric_limits<_Rhs>::max_exponent
                     && numeric_limits<_Lhs>::digits >= numeric_limits<_Rhs>::digits
                     && (numeric_limits<_Lhs>::is_signed || !numeric_limits<_Rhs>::is_signed))
  {
    return __fp_conv_rank_order::__greater;
  }
  else if constexpr (numeric_limits<_Lhs>::min_exponent >= numeric_limits<_Rhs>::min_exponent
                     && numeric_limits<_Lhs>::max_exponent <= numeric_limits<_Rhs>::max_exponent
                     && numeric_limits<_Lhs>::digits <= numeric_limits<_Rhs>::digits
                     && (!numeric_limits<_Lhs>::is_signed || numeric_limits<_Rhs>::is_signed))
  {
    return __fp_conv_rank_order::__less;
  }
  else
  {
    return __fp_conv_rank_order::__unordered;
  }
}

template <class _Lhs, class _Rhs>
_CCCL_INLINE_VAR constexpr __fp_conv_rank_order __fp_conv_rank_order_v = __fp_conv_rank_order_v_impl<_Lhs, _Rhs>();

template <class _Lhs, class _Rhs>
_CCCL_INLINE_VAR constexpr __fp_conv_rank_order __fp_conv_rank_order_int_ext_v =
  __fp_conv_rank_order_v<conditional_t<_CCCL_TRAIT(is_integral, _Lhs), double, _Lhs>,
                         conditional_t<_CCCL_TRAIT(is_integral, _Rhs), double, _Rhs>>;

template <class _Lhs, class _Rhs>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __fp_is_implicit_conversion_v_impl() noexcept
{
  constexpr auto __order = __fp_conv_rank_order_v<_Lhs, _Rhs>();
  return __order == __fp_conv_rank_order::__greater || __order == __fp_conv_rank_order::__equal;
}

template <class _Lhs, class _Rhs>
_CCCL_INLINE_VAR constexpr bool __fp_is_implicit_conversion_v = __fp_is_implicit_conversion_v_impl<_Lhs, _Rhs>();

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FLOATING_POINT_CONVERSION_RANK_ORDER_H
