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

#include <cuda/__cmath/uabs.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_add_overflow) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ADD_OVERFLOW(...) __builtin_add_overflow(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_add_overflow)

// nvc++ doesn't support 128-bit integers and crashes when certain type combinations are used (nvbug 5730860), so let's
// just disable the builtin for now.
#if _CCCL_COMPILER(NVHPC)
#  undef _CCCL_BUILTIN_ADD_OVERFLOW
#endif // _CCCL_COMPILER(NVHPC)

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __add_overflow_generic_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  using _Up  = ::cuda::std::make_unsigned_t<_Tp>;
  auto __sum = static_cast<_Tp>(static_cast<_Up>(__lhs) + static_cast<_Up>(__rhs));
  if constexpr (::cuda::std::is_signed_v<_Tp>)
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
  if constexpr ((sizeof(_Tp) == 4 || sizeof(_Tp) == 8) && ::cuda::std::is_unsigned_v<_Tp>)
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
#    if _CCCL_COMPILER(MSVC, >=, 19, 37)
    if constexpr (::cuda::std::is_signed_v<_Tp>)
    {
      overflow_result<_Tp> __result;
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
      return __result;
    }
    else
#    endif // _CCCL_COMPILER(MSVC, >=, 19, 37)
      if constexpr (::cuda::std::is_unsigned_v<_Tp>)
      { // unsigned
        overflow_result<_Tp> __result;
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
        return __result;
      }
      else
      {
        return ::cuda::__add_overflow_generic_impl(__lhs, __rhs);
      }
  }
  else
#  endif // ^^^ _CCCL_COMPILER(MSVC) || _CCCL_ARCH(X86_64) ^^^
  {
    return ::cuda::__add_overflow_generic_impl(__lhs, __rhs);
  }
}

#endif // _CCCL_HOST_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __add_overflow_uniform_type(_Tp __lhs, _Tp __rhs) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (return ::cuda::__add_overflow_device(__lhs, __rhs);),
                 (return ::cuda::__add_overflow_host(__lhs, __rhs);))
  }
  return ::cuda::__add_overflow_generic_impl(__lhs, __rhs);
}

template <typename _Result, typename _Lhs, typename _Rhs>
inline constexpr bool __is_add_representable_v =
  sizeof(_Result) > sizeof(_Lhs) && sizeof(_Result) > sizeof(_Rhs)
  && (::cuda::std::is_signed_v<_Result>
      || (::cuda::std::is_unsigned_v<_Lhs> && ::cuda::std::is_unsigned_v<_Rhs> && ::cuda::std::is_unsigned_v<_Result>) );

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
_CCCL_API constexpr overflow_result<_ActualResult> add_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  // We want to use __builtin_add_overflow only in host code. When compiling CUDA source file, we cannot use it in
  // constant expressions, because it doesn't work before nvcc 13.1 and is buggy in 13.1. When compiling C++ source
  // file, we can use it all the time.
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
#  if _CCCL_CUDA_COMPILATION()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
#  endif // _CCCL_CUDA_COMPILATION()
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   overflow_result<_ActualResult> __result{};
                   __result.overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__lhs, __rhs, &__result.value);
                   return __result;
                 }))
  }
#endif // _CCCL_BUILTIN_ADD_OVERFLOW

  // Host fallback + device implementation.
#if _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_ADD_OVERFLOW)
  using ::cuda::std::__make_nbit_int_t;
  using ::cuda::std::__make_nbit_uint_t;
  using ::cuda::std::__num_bits_v;
  using ::cuda::std::is_same_v;
  using ::cuda::std::is_signed_v;
  using ::cuda::std::is_unsigned_v;
  using _CommonAll                             = ::cuda::std::common_type_t<_Common, _ActualResult>;
  [[maybe_unused]] const bool __is_lhs_ge_zero = is_unsigned_v<_Lhs> || __lhs >= 0;
  [[maybe_unused]] const bool __is_rhs_ge_zero = is_unsigned_v<_Rhs> || __rhs >= 0;
  // shortcut for the case where inputs are representable with the max type
  if constexpr (__is_add_representable_v<_ActualResult, _Lhs, _Rhs>)
  {
    const auto __lhs1 = static_cast<_CommonAll>(__lhs);
    const auto __rhs1 = static_cast<_CommonAll>(__rhs);
    const auto __sum  = static_cast<_CommonAll>(__lhs1 + __rhs1);
    return ::cuda::overflow_cast<_ActualResult>(__sum);
  }
  // * int + int -> int
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs> && is_signed_v<_ActualResult>) // all signed
  {
    using _Sp         = __make_nbit_int_t<__num_bits_v<_CommonAll>>;
    const auto __lhs1 = static_cast<_Sp>(__lhs);
    const auto __rhs1 = static_cast<_Sp>(__rhs);
    const auto __sum  = ::cuda::__add_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret  = ::cuda::overflow_cast<_ActualResult>(__sum.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __sum.overflow};
  }
  // Positive inputs
  // * unsigned + unsigned (compile-time)
  // * unsigned + int >= 0 (compile-time + run-time check)
  // * int >= 0 + unsigned (compile-time + run-time check)
  // * int >= 0 + int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (__is_lhs_ge_zero && __is_rhs_ge_zero)
  {
    using _Up         = __make_nbit_uint_t<__num_bits_v<_CommonAll>>;
    const auto __lhs1 = static_cast<_Up>(__lhs);
    const auto __rhs1 = static_cast<_Up>(__rhs);
    const auto __sum  = ::cuda::__add_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret  = ::cuda::overflow_cast<_ActualResult>(__sum.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __sum.overflow};
  }
  // Negative inputs
  // * int < 0 + int < 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (!__is_lhs_ge_zero && !__is_rhs_ge_zero)
  {
    const auto __lhs1 = static_cast<_ActualResult>(__lhs);
    const auto __rhs1 = static_cast<_ActualResult>(__rhs);
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__lhs1 + __rhs1), true};
  }
  // Opposite signs
  // * int < 0 + int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above)
  // * int >= 0 + int < 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above)
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs>)
  {
    return ::cuda::overflow_cast<_ActualResult>(static_cast<_Common>(__lhs) + static_cast<_Common>(__rhs));
  }
  // Opposite signs
  // * unsigned + int < 0
  // * int < 0 + unsigned
  else
  {
    // skip checks in cmp_less, cmp_greater, uabs
    if constexpr (is_unsigned_v<_Lhs> && is_signed_v<_Lhs>)
    {
      _CCCL_ASSUME(__rhs < 0);
    }
    else if constexpr (is_unsigned_v<_Rhs> && is_signed_v<_Rhs>)
    {
      _CCCL_ASSUME(__lhs < 0);
    }
    const auto __lhs1 = static_cast<_CommonAll>(__lhs);
    const auto __rhs1 = static_cast<_CommonAll>(__rhs);
    const auto __sum  = static_cast<_CommonAll>(__lhs1 + __rhs1); // no overflow because of opposite signs
    // check if lhs + rhs is < 0,  e.g. lhs >= 0 && lhs < |rhs|
    if ((is_unsigned_v<_Lhs> && ::cuda::std::cmp_less(__lhs, ::cuda::uabs(__rhs)))
        || (is_unsigned_v<_Rhs> && ::cuda::std::cmp_greater(::cuda::uabs(__lhs), __rhs)))
    {
      if constexpr (is_unsigned_v<_ActualResult>)
      {
        return overflow_result<_ActualResult>{static_cast<_ActualResult>(__sum), true};
      }
      else
      {
        using _Sp = ::cuda::std::make_signed_t<_Common>;
        return ::cuda::overflow_cast<_ActualResult>(static_cast<_Sp>(__sum));
      }
    }
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__sum), false}; // because of opposite signs
  }
#endif // _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_ADD_OVERFLOW)
}

//! @brief Adds two numbers \p __lhs and \p __rhs with overflow detection
_CCCL_TEMPLATE(typename _Result, typename _Lhs, typename _Rhs)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Result> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs>
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr bool add_overflow(_Result& __result, const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  const auto __res = ::cuda::add_overflow<_Result>(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_ADD_OVERFLOW_H
