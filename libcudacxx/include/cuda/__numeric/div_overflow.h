//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_DIV_OVERFLOW_H
#define _CUDA___NUMERIC_DIV_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/neg.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Result, typename _Lhs, typename _Rhs>
inline constexpr bool __is_div_representable_v =
  (sizeof(_Result) > sizeof(_Lhs) && sizeof(_Result) > sizeof(_Rhs) && ::cuda::std::is_signed_v<_Result>)
  || (sizeof(_Result) >= sizeof(_Lhs) && sizeof(_Result) >= sizeof(_Rhs)
      && ::cuda::std::is_unsigned_v<_Lhs> && ::cuda::std::is_unsigned_v<_Rhs> && ::cuda::std::is_unsigned_v<_Result>);

/***********************************************************************************************************************
 * Public interface
 **********************************************************************************************************************/

_CCCL_TEMPLATE(typename _Result = void,
               typename _Lhs,
               typename _Rhs,
               typename _Common       = ::cuda::std::common_type_t<_Lhs, _Rhs>,
               typename _ActualResult = ::cuda::std::conditional_t<::cuda::std::is_void_v<_Result>, _Common, _Result>)
_CCCL_REQUIRES((::cuda::std::is_void_v<_Result> || ::cuda::std::__cccl_is_integer_v<_Result>)
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]]
_CCCL_API constexpr overflow_result<_ActualResult> div_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  if (__rhs == _Rhs{0})
  {
    return overflow_result<_ActualResult>{_ActualResult{}, true};
  }
  // the result is representable with the actual result type
  if constexpr (__is_div_representable_v<_ActualResult, _Lhs, _Rhs>)
  {
    const auto __lhs1   = static_cast<_ActualResult>(__lhs);
    const auto __rhs1   = static_cast<_ActualResult>(__rhs);
    const auto __result = static_cast<_ActualResult>(__lhs1 / __rhs1);
    return overflow_result<_ActualResult>{__result, false};
  }
  else
  {
    using ::cuda::std::is_same_v;
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    constexpr bool __both_signed = is_signed_v<_Lhs> && is_signed_v<_Rhs>;
    // special case for min / -1 -> potential overflow
    // -> the result is always smaller than lhs
    // unsigned result
    // - lhs < 0  && rhs >= 0 -> underflow
    // - lhs >= 0 && rhs < 0  -> underflow
    // - lhs < 0  && rhs < 0  -> ok, e.g. -7 / -2 = 3
    // - lhs >= 0 && rhs >= 0 -> ok
    // if constexpr (is_unsigned_v<_ActualResult>)
    //{
    bool __lhs_ge_zero = is_unsigned_v<_Lhs> || __lhs >= _Lhs{0};
    bool __rhs_ge_zero = is_unsigned_v<_Rhs> || __rhs >= _Rhs{0};

    if constexpr (__both_signed)
    {
      using _UnsignedLhs           = ::cuda::std::make_unsigned_t<_Lhs>;
      constexpr auto __lhs_min     = ::cuda::std::numeric_limits<_Lhs>::min();
      constexpr auto __neg_lhs_min = static_cast<_UnsignedLhs>(::cuda::neg(__lhs_min));
      if (__both_signed && __lhs == __lhs_min && __rhs == _Rhs{-1})
      {
        constexpr auto __result_max = ::cuda::std::numeric_limits<_ActualResult>::max();
        if constexpr (::cuda::std::cmp_greater(__neg_lhs_min, __result_max))
        {
          return overflow_result<_ActualResult>{_ActualResult{}, true};
        }
        else
        {
          return overflow_result<_ActualResult>{_ActualResult{__neg_lhs_min}, false};
        }
      }
      using _CommonAll    = ::cuda::std::common_type_t<_Common, _ActualResult>;
      using _SignedResult = ::cuda::std::make_signed_t<_CommonAll>;
      auto __lhs1         = static_cast<_SignedResult>(__lhs);
      auto __rhs1         = static_cast<_SignedResult>(__rhs);
      return ::cuda::overflow_cast<_ActualResult>(__lhs1 / __rhs1);
    }
    else if (__lhs_ge_zero && __rhs_ge_zero)
    {
      using _CommonAll      = ::cuda::std::common_type_t<_Common, _ActualResult>;
      using _UnsignedResult = ::cuda::std::make_unsigned_t<_CommonAll>;
      auto __lhs1           = static_cast<_UnsignedResult>(__lhs);
      auto __rhs1           = static_cast<_UnsignedResult>(__rhs);
      return ::cuda::overflow_cast<_ActualResult>(__lhs1 / __rhs1);
    }
    else
    {
      if constexpr (is_unsigned_v<_ActualResult>)
      {
        return overflow_result<_ActualResult>{_ActualResult{}, true};
      }
      else
      {
        auto __lhs1 = ::cuda::uabs(__lhs);
        auto __rhs1 = ::cuda::uabs(__rhs);
        auto __ret  = ::cuda::overflow_cast<_ActualResult>(__lhs1 / __rhs1);
        return overflow_result<_ActualResult>{static_cast<_ActualResult>(-__ret.value), __ret.overflow};
      }
    }
    //}
    // else // signed result
    //{
    //  return ::cuda::overflow_cast<_ActualResult>(__lhs / __rhs);
    //}
  }
}

//! @brief Divides two numbers \p __lhs and \p __rhs with overflow detection
_CCCL_TEMPLATE(typename _Result, typename _Lhs, typename _Rhs)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Result> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs>
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr bool div_overflow(_Result& __result, const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  const auto __res = ::cuda::div_overflow<_Result>(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_DIV_OVERFLOW_H
