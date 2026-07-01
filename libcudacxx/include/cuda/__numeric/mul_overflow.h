//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_MUL_OVERFLOW_H
#define _CUDA___NUMERIC_MUL_OVERFLOW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/mul_hi.h>
#include <cuda/__cmath/neg.h>
#include <cuda/__cmath/uabs.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/__numeric/overflow_result.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_mul_overflow) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_MUL_OVERFLOW(...) __builtin_mul_overflow(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_mul_overflow) || _CCCL_COMPILER(GCC)

// nvc++ < 26.1 doesn't support 128-bit integers and crashes when certain type combinations are used (nvbug 5730860).
#if _CCCL_COMPILER(NVHPC, <, 26, 1)
#  undef _CCCL_BUILTIN_MUL_OVERFLOW
#endif // _CCCL_COMPILER(NVHPC, <, 26, 1)

// On ARM64, using the builtin with 128-bit ints result in `undefined reference to __muloti4` with nvc++ and clang < 20.
#if _CCCL_ARCH(ARM64) && (_CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG, <, 20))
#  undef _CCCL_BUILTIN_MUL_OVERFLOW
#endif // _CCCL_ARCH(ARM64) && (_CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG, <, 20))

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Result, class _Lhs, class _Rhs>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Result> __mul_overflow_generic(_Lhs __lhs, _Rhs __rhs) noexcept
{
  using ::cuda::std::__cccl_uintmax_t;
  using ::cuda::std::__num_bits_v;
  using ::cuda::std::is_signed_v;
  using ::cuda::std::make_unsigned_t;

  using _UResult = make_unsigned_t<_Result>;
  using _ULhs    = make_unsigned_t<_Lhs>;
  using _URhs    = make_unsigned_t<_Rhs>;

  const auto __uabs_lhs        = static_cast<_ULhs>(::cuda::uabs(__lhs));
  const auto __uabs_rhs        = static_cast<_URhs>(::cuda::uabs(__rhs));
  const auto __negative_result = (::cuda::std::cmp_greater_equal(__lhs, 0) != ::cuda::std::cmp_greater_equal(__rhs, 0));

  auto __overflow_mul = false;

  if (__negative_result && !is_signed_v<_Result>)
  {
    __overflow_mul = true;
  }
  else
  {
    constexpr auto __nbits = __num_bits_v<_Result>;
    const auto c           = is_signed_v<_Result>
                             ? (static_cast<_UResult>(1) << (__nbits - 1)) - static_cast<_UResult>(__negative_result ? 0 : 1)
                             : static_cast<_UResult>(-1);

    if (__uabs_rhs != 0)
    {
      __overflow_mul = __uabs_lhs > c / __uabs_rhs;
    }
    else if (__uabs_lhs != 0)
    {
      __overflow_mul = __uabs_rhs > c / __uabs_lhs;
    }
  }

  // Floor to 16; smallest register size is 16-bits
  constexpr auto __max_nbits = ::cuda::std::max({__num_bits_v<_Lhs>, __num_bits_v<_Rhs>, 16});
  using _Up                  = ::cuda::std::__make_nbit_int_t<__max_nbits, false>;
  _Up __result_lo            = 0;
  _Up __ulhs                 = static_cast<_Up>(__lhs);
  _Up __urhs                 = static_cast<_Up>(__rhs);
  if constexpr ((__max_nbits <= sizeof(int16_t)))
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT{
      NV_IF_TARGET(NV_IS_DEVICE, ({ asm("mul.lo.u16 %0, %1, %2;"
                                        : "=h"(__result_lo)
                                        : "h"(__ulhs), "h"(__urhs)); }))};
    return {static_cast<_Result>(__result_lo), __overflow_mul};
  }
  else if constexpr ((__max_nbits <= sizeof(int32_t)))
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT{
      NV_IF_TARGET(NV_IS_DEVICE, ({ asm("mul.lo.u32 %0, %1, %2;"
                                        : "=r"(__result_lo)
                                        : "r"(__ulhs), "r"(__urhs)); }))};
    return {static_cast<_Result>(__result_lo), __overflow_mul};
  }
  else if constexpr ((__max_nbits <= sizeof(int64_t)))
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT{
      NV_IF_TARGET(NV_IS_DEVICE, ({ asm("mul.lo.u64 %0, %1, %2;"
                                        : "=l"(__result_lo)
                                        : "l"(__ulhs), "l"(__urhs)); }))};
    return {static_cast<_Result>(__result_lo), __overflow_mul};
  }
  else
  {
    // Multiply with overflow results in UB, which isn't allowed during constant
    // evaluation. Instead, multiply the absolute values of lhs and rhs and
    // apply negative sign if needed to get expected low-word result
    const auto __uresult_lo = __cccl_uintmax_t{__uabs_lhs} * __cccl_uintmax_t{__uabs_rhs};
    const auto __result_lo  = static_cast<_Result>((__negative_result) ? ::cuda::neg(__uresult_lo) : __uresult_lo);
    return {__result_lo, __overflow_mul};
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API overflow_result<_Tp> __mul_overflow_host(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int8_t))
    {
      ::cuda::std::int16_t __result;
      bool __overflow = ::_mul_full_overflow_i8(__lhs, __rhs, &__result);
      return {static_cast<_Tp>(__result), __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int16_t))
    {
      ::cuda::std::int16_t __result;
      bool __overflow = ::_mul_overflow_i16(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int32_t))
    {
      ::cuda::std::int32_t __result;
      bool __overflow = ::_mul_overflow_i32(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::int64_t))
    {
      ::cuda::std::int64_t __result;
      bool __overflow = ::_mul_overflow_i64(__lhs, __rhs, &__result);
      return {__result, __overflow};
    }
    else
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    {
      return ::cuda::__mul_overflow_generic<_Tp>(__lhs, __rhs);
    }
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint8_t))
    {
      ::cuda::std::uint16_t __result;
      bool __overflow = ::_mul_full_overflow_u8(__lhs, __rhs, &__result);
      return {static_cast<_Tp>(__result), __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint16_t))
    {
      ::cuda::std::uint16_t __lo;
      ::cuda::std::uint16_t __hi;
      bool __overflow = ::_mul_full_overflow_u16(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint32_t))
    {
      ::cuda::std::uint32_t __lo;
      ::cuda::std::uint32_t __hi;
      bool __overflow = ::_mul_full_overflow_u32(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint64_t))
    {
      ::cuda::std::uint64_t __lo;
      ::cuda::std::uint64_t __hi;
      bool __overflow = ::_mul_full_overflow_u64(__lhs, __rhs, &__lo, &__hi);
      return {__lo, __overflow};
    }
    else
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    {
      return ::cuda::__mul_overflow_generic<_Tp>(__lhs, __rhs);
    }
  } // ^^^ unsigned types ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_TEMPLATE(class _Result = void,
               class _Lhs,
               class _Rhs,
               class _Common    = ::cuda::std::common_type_t<_Lhs, _Rhs>,
               class _ActResult = ::cuda::std::conditional_t<::cuda::std::is_void_v<_Result>, _Common, _Result>)
_CCCL_REQUIRES((::cuda::std::is_void_v<_Result> || ::cuda::std::__cccl_is_integer_v<_Result>)
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr overflow_result<_ActResult> mul_overflow(_Lhs __lhs, _Rhs __rhs) noexcept
{
  // We want to use __builtin_mul_overflow only in host code. When compiling CUDA source file, we cannot use it in
  // constant expressions, because it doesn't work before nvcc 13.1 and is buggy in 13.1. When compiling C++ source
  // file, we can use it all the time.
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
#  if _CCCL_CUDA_COMPILATION()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
#  endif // _CCCL_CUDA_COMPILATION()
  {
    // nvc++ doesn't fully support 128-bit ints with __builtin_mul_overflow.
#  if _CCCL_COMPILER(NVHPC)
    if constexpr (sizeof(_ActResult) != 16 && sizeof(_Lhs) != 16 && sizeof(_Rhs) != 16)
#  endif // _CCCL_COMPILER(NVHPC)
    {
      NV_IF_TARGET(NV_IS_HOST, ({
                     overflow_result<_ActResult> __result{};
                     __result.overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__lhs, __rhs, &__result.value);
                     return __result;
                   }))
    }
  }
#endif // _CCCL_BUILTIN_MUL_OVERFLOW

  // Host fallback + device implementation.
#if _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_MUL_OVERFLOW) || (_CCCL_HAS_INT128() && _CCCL_COMPILER(NVHPC))
  using ::cuda::std::is_signed_v;

  // If we would check for is_same_v, we would get slow path for e. g. long and long long, even though they represent
  // the same range.
  constexpr auto __all_same_size = sizeof(_ActResult) == sizeof(_Lhs) && sizeof(_ActResult) == sizeof(_Rhs);
  constexpr auto __all_same_sign =
    is_signed_v<_ActResult> == is_signed_v<_Lhs> && is_signed_v<_ActResult> == is_signed_v<_Rhs>;
  if constexpr (__all_same_size && __all_same_sign)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      NV_IF_TARGET(
        NV_IS_HOST,
        (return ::cuda::__mul_overflow_host(static_cast<_ActResult>(__lhs), static_cast<_ActResult>(__rhs));))
    }
  }
  return ::cuda::__mul_overflow_generic<_ActResult>(__lhs, __rhs);
#endif // needs fallback
}

_CCCL_TEMPLATE(class _Result, class _Lhs, class _Rhs)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Result> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs>
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr bool mul_overflow(_Result& __result, _Lhs __lhs, _Rhs __rhs) noexcept
{
  const auto __overflow_result = ::cuda::mul_overflow<_Result>(__lhs, __rhs);
  __result                     = __overflow_result.value;
  return __overflow_result.overflow;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_MUL_OVERFLOW_H
