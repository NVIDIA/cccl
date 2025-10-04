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
  if constexpr (__is_div_representable_v<_ActualResult, _Lhs, _Rhs>)
  {
    const auto __div = static_cast<_ActualResult>(__lhs / __rhs);
    return overflow_result<_ActualResult>{__div, false};
  }
  else
  {
    using ::cuda::std::is_same_v;
    using ::cuda::std::is_signed_v;
    using ::cuda::std::is_unsigned_v;
    // special case for -1 / min -> overflow
    if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs> && sizeof(_Lhs) >= sizeof(_ActualResult))
    {
      constexpr auto __lhs_min = ::cuda::std::numeric_limits<_Lhs>::min();
      if (__lhs == __lhs_min && __rhs == _Rhs{-1})
      {
        if constexpr (sizeof(_ActualResult) <= sizeof(_Lhs) && (is_signed_v<_ActualResult>)
                      || sizeof(_ActualResult) < sizeof(_Lhs) && is_unsigned_v<_ActualResult>)
        {
          return overflow_result<_ActualResult>{_ActualResult{}, true};
        }
        else
        {
          constexpr auto __result = static_cast<_ActualResult>(::cuda::neg(__lhs_min));
          return overflow_result<_ActualResult>{__result, false};
        }
      }
    }
    // e.g. 7 / -2 = unsigned -> underflow
    auto __ret = ::cuda::overflow_cast<_ActualResult>(__lhs / __rhs);
    if constexpr (is_unsigned_v<_ActualResult>)
    {
      bool __lhs_less_than_zero = !is_unsigned_v<_Lhs> && __lhs < _Lhs{0};
      bool __rhs_less_than_zero = !is_unsigned_v<_Rhs> && __rhs < _Rhs{0};
      bool __overflow = (__lhs > _Lhs{0} && __rhs_less_than_zero) || (__lhs_less_than_zero && !__rhs_less_than_zero);
      return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __overflow};
    }
    return __ret;
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
