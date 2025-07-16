//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_ADD_OVERFLOW_H
#define _CUDA___NUMERIC_ADD_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __add_overflow_generic_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  using _Up  = _CUDA_VSTD::make_unsigned_t<_Tp>;
  auto __sum = static_cast<_Tp>(static_cast<_Up>(__lhs) + static_cast<_Up>(__rhs));
  if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
  {
    return overflow_result<_Tp>{__sum, (__sum < __lhs) == (__rhs >= _Tp{0})};
  }
  else
  {
    return overflow_result<_Tp>{__sum, __sum < __lhs};
  }
}
#if _CCCL_DEVICE_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API overflow_result<_Tp> __add_overflow_device(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr ((sizeof(_Tp) == 4 || sizeof(_Tp) == 8) && _CUDA_VSTD::is_unsigned_v<_Tp>)
  {
    _Tp __result;
    int __overflow;
    if constexpr (sizeof(_Tp) == 4)
    {
      asm("add.cc.u32 %0, %2, %3;"
          "addc.u32 %1, 0, 0;"
          : "=r"(__result), "=r"(__overflow)
          : "r"(__lhs), "r"(__rhs));
    }
    else if constexpr (sizeof(_Tp) == 8)
    {
      asm("add.cc.u64 %0, %2, %3;"
          "addc.u32 %1, 0, 0;"
          : "=l"(__result), "=r"(__overflow)
          : "l"(__lhs), "l"(__rhs));
    }
    return overflow_result<_Tp>{__result, static_cast<bool>(__overflow)};
  }
  else
  {
    return ::cuda::__add_overflow_generic_impl(__lhs, __rhs); // do not use builtin functions
  }
}

#endif // _CCCL_DEVICE_COMPILATION()

#if _CCCL_HOST_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API overflow_result<_Tp> __add_overflow_host(_Tp __lhs, _Tp __rhs) noexcept
{
#  if _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
  if constexpr (sizeof(_Tp) <= 8)
  {
    overflow_result<_Tp> __result;
#    if _CCCL_COMPILER(MSVC, >=, 19, 37)
    if constexpr (_CUDA_VSTD::is_signed_v<_Tp>)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        __result.overflow = ::_add_overflow_i8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = ::_add_overflow_i16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = ::_add_overflow_i32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = ::_add_overflow_i64(0, __lhs, __rhs, &__result.value);
      }
    }
    else
#    endif // _CCCL_COMPILER(MSVC, >=, 19, 37)
    { // unsigned
      if constexpr (sizeof(_Tp) == 1)
      {
        __result.overflow = ::_addcarry_u8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = ::_addcarry_u16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = ::_addcarry_u32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = ::_addcarry_u64(0, __lhs, __rhs, &__result.value);
      }
    }
    return __result;
  }
#  else
  return ::cuda::__add_overflow_generic_impl(__lhs, __rhs);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
}

#endif // _CCCL_HOST_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __add_overflow_dispatch(_Tp __lhs, _Tp __rhs) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (return ::cuda::__add_overflow_device(__lhs, __rhs);),
                 (return ::cuda::__add_overflow_host(__lhs, __rhs);))
  }
  return ::cuda::__add_overflow_generic_impl(__lhs, __rhs);
}

/***********************************************************************************************************************
 * Public interface
 **********************************************************************************************************************/

//! @brief Adds two numbers \p __lhs and \p __rhs with overflow detection
_CCCL_TEMPLATE(
  typename _Result = void,
  typename _Lhs,
  typename _Rhs,
  typename _ActualResult =
    _CUDA_VSTD::conditional_t<_CUDA_VSTD::is_void_v<_Result>, _CUDA_VSTD::common_type_t<_Lhs, _Rhs>, _Result>)
_CCCL_REQUIRES((_CUDA_VSTD::is_void_v<_Result> || _CUDA_VSTD::__cccl_is_integer_v<_Result>)
                 _CCCL_AND _CUDA_VSTD::__cccl_is_integer_v<_Lhs> _CCCL_AND _CUDA_VSTD::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr overflow_result<_ActualResult>
add_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW) && !_CCCL_CUDA_COMPILER(NVCC)
  overflow_result<_Tp> __result;
  __result.overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__lhs, __rhs, &__result.value);
  return __result;
#else
  using _CUDA_VSTD::is_signed_v;
  using _CUDA_VSTD::is_unsigned_v;
  using _Common     = _CUDA_VSTD::common_type_t<_Lhs, _Rhs>;
  const auto __lhs1 = static_cast<_ActualResult>(__lhs);
  const auto __rhs1 = static_cast<_ActualResult>(__rhs);
  // no overflow
  if constexpr (sizeof(_ActualResult) > sizeof(_Lhs) && sizeof(_ActualResult) > sizeof(_Rhs)
                && (is_signed_v<_ActualResult> || (is_unsigned_v<_Lhs> && is_unsigned_v<_Rhs>) ))
  {
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__lhs1 + __rhs1), false};
  }
  // uncommon case
  else if constexpr (sizeof(_ActualResult) < sizeof(_Common))
  {
    auto __ret      = ::cuda::add_overflow(__lhs, __rhs); // perform the computation at higher precision
    auto __ret_cast = ::cuda::overflow_cast<_ActualResult>(__ret.value);
    return overflow_result<_ActualResult>{__ret_cast.value, __ret.overflow || __ret_cast.overflow};
  }
  // uncommon case
  // else if constexpr (sizeof(_ActualResult) == sizeof(_Common) && is_signed_v<_ActualResult> &&
  // is_unsigned_v<_Common>)
  //{
  //  return ::cuda::add_overflow(static_cast<_ActualResult>(__lhs), static_cast<_ActualResult>(__rhs));
  //}
  else
  {
    const bool __is_lhs_lt_zero = is_signed_v<_Lhs> && __lhs < 0;
    const bool __is_rhs_lt_zero = is_signed_v<_Rhs> && __rhs < 0;
    const int __lhs_rhs_sign    = __is_lhs_lt_zero + __is_rhs_lt_zero;
    // potential underflow
    if (is_unsigned_v<_ActualResult> && __lhs_rhs_sign)
    {
      auto __sum = static_cast<_ActualResult>(__lhs1 + __rhs1);
      using _Sp  = _CUDA_VSTD::make_signed_t<_ActualResult>;
      return overflow_result<_ActualResult>{__sum, __lhs_rhs_sign == 2 || static_cast<_Sp>(__sum) < _Sp{0}};
    }
    if constexpr (is_signed_v<_ActualResult>) // suppress clang-14 warning
    {
      if constexpr (is_unsigned_v<_Rhs> && sizeof(_ActualResult) > sizeof(_Rhs))
      {
        _CCCL_ASSUME(__rhs1 >= 0); // skip two comparisons
      }
      else if constexpr (is_unsigned_v<_Lhs> && sizeof(_ActualResult) > sizeof(_Lhs))
      {
        _CCCL_ASSUME(__lhs1 >= 0);
        return ::cuda::__add_overflow_dispatch(__rhs1, __lhs1); // skip two comparisons
      }
    }
    return ::cuda::__add_overflow_dispatch(__lhs1, __rhs1);
  }
#endif // defined(_CCCL_BUILTIN_ADD_OVERFLOW) && !_CCCL_CUDA_COMPILER(NVCC)
}

//! @brief Adds two numbers \p __lhs and \p __rhs with overflow detection
_CCCL_TEMPLATE(typename _Result, typename _Lhs, typename _Rhs)
_CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_integer_v<_Result> _CCCL_AND _CUDA_VSTD::__cccl_is_integer_v<_Lhs> _CCCL_AND
                 _CUDA_VSTD::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr bool add_overflow(_Result& __result, const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  const auto __res = ::cuda::add_overflow<_Result>(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_ADD_OVERFLOW_H
