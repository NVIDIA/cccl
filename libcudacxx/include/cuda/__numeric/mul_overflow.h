//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MUL_OVERFLOW_H
#define _CUDA___MUL_OVERFLOW_H

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
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/climits>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_Tp>
__mul_overflow_constexpr(_Tp __lhs, _Tp __rhs) noexcept
{
  // If there is a wider type available, upcast the operands and check for overflow
  if constexpr (sizeof(_Tp) < sizeof(_CUDA_VSTD::__cccl_intmax_t))
  {
    using _Up               = _CUDA_VSTD::__make_nbit_int_t<sizeof(_Tp) * CHAR_BIT * 2>;
    const auto __mul_result = static_cast<_Up>(__lhs) * static_cast<_Up>(__rhs);

    overflow_result<_Tp> __result{};
    __result.value    = static_cast<_Tp>(__mul_result);
    __result.overflow = !_CUDA_VSTD::in_range<_Tp>(__mul_result);
    return __result;
  }
  else if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
    const auto __lhs_sign = _CUDA_VSTD::cmp_less(__lhs, 0);
    const auto __rhs_sign = _CUDA_VSTD::cmp_less(__rhs, 0);

    const auto __lhs_abs = ::cuda::uabs(__lhs);
    const auto __rhs_abs = ::cuda::uabs(__rhs);

    const auto __result_sign = __lhs_sign ^ __rhs_sign;
    const auto __result_max =
      ::cuda::uabs((__result_sign) ? _CUDA_VSTD::numeric_limits<_Tp>::min() : _CUDA_VSTD::numeric_limits<_Tp>::max());

    const auto __mul_result = ::cuda::__mul_overflow_constexpr(__lhs_abs, __rhs_abs);

    overflow_result<_Tp> __result{};
    __result.value    = static_cast<_Tp>((__result_sign) ? ::cuda::__neg(__mul_result.value) : __mul_result.value);
    __result.overflow = __mul_result.overflow || _CUDA_VSTD::cmp_greater(__mul_result.value, __result_max);
    return __result;
  }
  else
  {
    overflow_result<_Tp> __result{};
    __result.value    = __lhs * __rhs;
    __result.overflow = __lhs != _Tp(0) && __rhs != _Tp(0) && _CUDA_VSTD::numeric_limits<_Tp>::max() / __lhs < __rhs;
    return __result;
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HOST _CCCL_HIDE_FROM_ABI overflow_result<_Tp> __mul_overflow_host(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Tp))
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int8_t))
    {
      _CUDA_VSTD::int16_t __result;
      bool __overflow = ::_mul_full_overflow_i8(__lhs, __rhs, &__result);
      return {static_cast<_Tp>(__result), __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int16_t))
    {
      _CUDA_VSTD::int16_t __result;
      bool __overflow = ::_mul_overflow_i16(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int32_t))
    {
      _CUDA_VSTD::int32_t __result;
      bool __overflow = ::_mul_overflow_i32(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int64_t))
    {
      _CUDA_VSTD::int64_t __result;
      bool __overflow = ::_mul_overflow_i64(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else
    {
      return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
    }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int32_t))
    {
      const _CUDA_VSTD::int64_t __result = ::__emul(__lhs, __rhs);
      return {static_cast<_Tp>(__result), !_CUDA_VSTD::in_range<_Tp>(__result)};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::int64_t))
    {
      _CUDA_VSTD::int64_t __hi = ::__mulh(__lhs, __rhs);
      _CUDA_VSTD::int64_t __lo = __lhs * __rhs;
      return {__lo, __hi != _Tp{0} && __hi != _Tp{-1}};
    }
    else
    {
      return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint8_t))
    {
      _CUDA_VSTD::uint16_t __result;
      bool __overflow = ::_mul_full_overflow_u8(__lhs, __rhs, &__result);
      return {static_cast<_Tp>(__result), __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint16_t))
    {
      _CUDA_VSTD::uint16_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u16(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint32_t))
    {
      _CUDA_VSTD::uint32_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u32(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint64_t))
    {
      _CUDA_VSTD::uint64_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u64(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else
    {
      return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
    }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint32_t))
    {
      const _CUDA_VSTD::uint64_t __result = ::__emulu(__lhs, __rhs);
      return {static_cast<_Tp>(__result), !_CUDA_VSTD::in_range<_Tp>(__result)};
    }
    else if constexpr (sizeof(_Tp) == sizeof(_CUDA_VSTD::uint64_t))
    {
      return {__lhs * __rhs, ::__umulh(__lhs, __rhs) != _Tp{0}};
    }
    else
    {
      return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
  } // ^^^ unsigned types ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

// todo: optimize for device?
// template <class _Tp>
// [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI overflow_result<_Tp> __mul_overflow_device(_Tp __lhs, _Tp __rhs)
// noexcept
// {
//   return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
// }

_CCCL_TEMPLATE(
  class _Result = void,
  class _Lhs,
  class _Rhs,
  class _ActResult =
    _CUDA_VSTD::conditional_t<_CCCL_TRAIT(_CUDA_VSTD::is_void, _Result), _CUDA_VSTD::common_type_t<_Lhs, _Rhs>, _Result>)
_CCCL_REQUIRES((_CCCL_TRAIT(_CUDA_VSTD::is_void, _Result) || _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Result))
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Lhs)
                   _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr overflow_result<_ActResult>
mul_overflow(_Lhs __lhs, _Rhs __rhs) noexcept
{
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  overflow_result<_ActResult> __result{};
  __result.overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__lhs, __rhs, &__result.value);
  return __result;
#else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
  if constexpr (_CCCL_TRAIT(_CUDA_VSTD::is_same, _ActResult, _Lhs)
                && _CCCL_TRAIT(_CUDA_VSTD::is_same, _ActResult, _Rhs))
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, (return ::cuda::__mul_overflow_host(__lhs, __rhs);))
      // NV_IF_TARGET(NV_IS_DEVICE, (return ::cuda::__mul_overflow_device(__lhs, __rhs);))
    }
    return ::cuda::__mul_overflow_constexpr(__lhs, __rhs);
  }
  // fast path if the result type is at least twice the size of the operands, but we need to handle the case when the
  // result type is signed and both operands are unsigned
  else if constexpr (sizeof(_ActResult) >= 2 * sizeof(_Lhs) && sizeof(_ActResult) >= 2 * sizeof(_Rhs)
                     && ((_CCCL_TRAIT(_CUDA_VSTD::is_signed, _ActResult)
                          == (_CCCL_TRAIT(_CUDA_VSTD::is_signed, _Lhs) || _CCCL_TRAIT(_CUDA_VSTD::is_signed, _Rhs)))
                         || sizeof(_Lhs) != sizeof(_Rhs)))
  {
    return {static_cast<_ActResult>(_ActResult(__lhs) * _ActResult(__rhs)), false};
  }
  // generic slow path
  else
  {
    using _UCommon = _CUDA_VSTD::make_unsigned_t<_CUDA_VSTD::common_type_t<_Lhs, _Rhs, _ActResult>>;

    const auto __lhs_sign = _CUDA_VSTD::cmp_less(__lhs, 0);
    const auto __rhs_sign = _CUDA_VSTD::cmp_less(__rhs, 0);

    const auto __lhs_abs = static_cast<_UCommon>(::cuda::uabs(__lhs));
    const auto __rhs_abs = static_cast<_UCommon>(::cuda::uabs(__rhs));

    const auto __result_sign = __lhs_sign ^ __rhs_sign;
    const auto __result_max  = static_cast<_UCommon>(::cuda::uabs(
      (__result_sign) ? _CUDA_VSTD::numeric_limits<_ActResult>::min() : _CUDA_VSTD::numeric_limits<_ActResult>::max()));

    const auto __mul_result = ::cuda::mul_overflow<_UCommon>(__lhs_abs, __rhs_abs);

    overflow_result<_ActResult> __result{};
    __result.value = static_cast<_ActResult>((__result_sign) ? ::cuda::__neg(__mul_result.value) : __mul_result.value);
    __result.overflow = __mul_result.overflow || _CUDA_VSTD::cmp_greater(__mul_result.value, __result_max);
    return __result;
  }

#endif // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^
}

_CCCL_TEMPLATE(class _Result, class _Lhs, class _Rhs)
_CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _Result)
                 _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _Lhs)
                   _CCCL_AND _CCCL_TRAIT(_CUDA_VSTD::__cccl_is_cv_integer, _Rhs))
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool mul_overflow(_Result& __result, _Lhs __lhs, _Rhs __rhs) noexcept
{
  const auto __overflow_result = ::cuda::mul_overflow<_Result>(__lhs, __rhs);
  __result                     = __overflow_result.value;
  return __overflow_result.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MUL_OVERFLOW_H
