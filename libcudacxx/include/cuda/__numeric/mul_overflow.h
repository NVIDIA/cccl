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
#if _CCCL_HOST_ARCH(ARM64) && (_CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG, <, 20))
#  undef _CCCL_BUILTIN_MUL_OVERFLOW
#endif // _CCCL_HOST_ARCH(ARM64) && (_CCCL_COMPILER(NVHPC) || _CCCL_COMPILER(CLANG, <, 20))

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __mul_overflow_generic_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  using ::cuda::std::__num_bits_v;

  if constexpr (__num_bits_v<_Tp> < __num_bits_v<uint32_t>)
  {
    using _Up            = ::cuda::std::__make_nbit_int_t<__num_bits_v<uint32_t>, ::cuda::std::is_signed_v<_Tp>>;
    const auto __product = static_cast<_Up>(__lhs) * static_cast<_Up>(__rhs);
    return {static_cast<_Tp>(__product), !::cuda::std::in_range<_Tp>(__product)};
  }
  else if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
    using _Up               = ::cuda::std::make_unsigned_t<_Tp>;
    const auto __lhs1       = static_cast<_Up>(__lhs);
    const auto __rhs1       = static_cast<_Up>(__rhs);
    const auto __product_lo = static_cast<_Tp>(__lhs1 * __rhs1);
    auto __product_hi       = ::cuda::mul_hi(__lhs1, __rhs1);
    const auto __expected   = __product_lo < 0 ? static_cast<_Up>(-1) : static_cast<_Up>(0);
    if (__rhs < 0)
    {
      __product_hi -= __lhs1;
    }
    if (__lhs < 0)
    {
      __product_hi -= __rhs1;
    }
    return {__product_lo, __product_hi != __expected};
  }
  else
  {
    using _Up               = ::cuda::std::make_unsigned_t<_Tp>;
    const auto __lhs1       = static_cast<_Up>(__lhs);
    const auto __rhs1       = static_cast<_Up>(__rhs);
    const auto __product_lo = static_cast<_Tp>(__lhs1 * __rhs1);
    const auto __product_hi = ::cuda::mul_hi(__lhs1, __rhs1);
    return {__product_lo, __product_hi != 0};
  }
}

#if _CCCL_DEVICE_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API overflow_result<_Tp> __mul_overflow_device(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (::cuda::std::is_unsigned_v<_Tp>)
  {
    using ::cuda::std::uint32_t;
    using ::cuda::std::uint64_t;

    if constexpr (sizeof(_Tp) < sizeof(uint32_t))
    {
      const auto __result = uint32_t{__lhs} * uint32_t{__rhs};
      return {static_cast<_Tp>(__result), !::cuda::std::in_range<_Tp>(__result)};
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      uint32_t __result;
      uint32_t __hi;
      asm("mul.lo.u32 %0, %2, %3;"
          "mul.hi.u32 %1, %2, %3;"
          : "=r"(__result), "=r"(__hi)
          : "r"(__lhs), "r"(__rhs));
      return {__result, __hi != 0};
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      uint64_t __result;
      uint64_t __hi;
      asm("mul.lo.u64 %0, %2, %3;"
          "mul.hi.u64 %1, %2, %3;"
          : "=l"(__result), "=l"(__hi)
          : "l"(__lhs), "l"(__rhs));
      return {__result, __hi != 0};
    }
#  if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__uint128_t))
    {
      const uint64_t __a0 = static_cast<uint64_t>(__lhs);
      const uint64_t __a1 = static_cast<uint64_t>(__lhs >> 64);
      const uint64_t __b0 = static_cast<uint64_t>(__rhs);
      const uint64_t __b1 = static_cast<uint64_t>(__rhs >> 64);

      uint64_t __r0, __r1, __r2, __r3;
      asm("mul.lo.u64      %0, %4, %6;" // r0 = lo(a0 * b0)
          "mul.hi.u64      %1, %4, %6;" // r1 = hi(a0 * b0)
          "mad.lo.cc.u64   %1, %4, %7, %1;" // r1 += lo(a0 * b1)
          "madc.hi.u64     %2, %4, %7, 0;" // r2  = hi(a0 * b1) + carry
          "mad.lo.cc.u64   %1, %5, %6, %1;" // r1 += lo(a1 * b0)
          "madc.hi.cc.u64  %2, %5, %6, %2;" // r2 += hi(a1 * b0) + carry
          "addc.u64        %3, 0, 0;" // r3  = carry-out
          "mad.lo.cc.u64   %2, %5, %7, %2;" // r2 += lo(a1 * b1)
          "madc.hi.u64     %3, %5, %7, %3;" // r3 += hi(a1 * b1) + carry
          : "=l"(__r0), "=l"(__r1), "=l"(__r2), "=l"(__r3)
          : "l"(__a0), "l"(__a1), "l"(__b0), "l"(__b1));

      const auto __result   = (static_cast<__uint128_t>(__r1) << 64) | __r0;
      const bool __overflow = (__r2 != 0) || (__r3 != 0);
      return {__result, __overflow};
    }
#  endif // _CCCL_HAS_INT128()
    else
    {
      ::cuda::__mul_overflow_generic_impl(__lhs, __rhs); // do not use builtin functions
    }
  }
  else
  {
    using ::cuda::std::int32_t;

    if constexpr (sizeof(_Tp) < sizeof(int32_t))
    {
      const auto __product = int32_t{__lhs} * int32_t{__rhs};
      return {static_cast<_Tp>(__product), !::cuda::std::in_range<_Tp>(__product)};
    }
#  if _CCCL_HAS_INT128()
    else if constexpr (sizeof(_Tp) == sizeof(__int128_t))
    {
      using _Up                = ::cuda::std::make_unsigned_t<_Tp>;
      const auto __umul_result = ::cuda::__mul_overflow_device(static_cast<_Up>(__lhs), static_cast<_Up>(__rhs));
      const auto __result      = static_cast<_Tp>(__umul_result.value);
      const auto __overflow    = ((__lhs >= 0) == (__rhs >= 0)) && (__umul_result.overflow == (__result >= 0));
      return {__result, __overflow};
    }
#  endif // _CCCL_HAS_INT128()
    else
    {
      // For 32 and 64 bit ints, this seems to be the more efficient path.
      return ::cuda::__mul_overflow_generic_impl(__lhs, __rhs);
    }
  }
}

#endif // _CCCL_DEVICE_COMPILATION()

#if _CCCL_HOST_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API overflow_result<_Tp> __mul_overflow_host(_Tp __lhs, _Tp __rhs) noexcept
{
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_HOST_ARCH(X86_64)
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
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_HOST_ARCH(X86_64)
    {
      return ::cuda::__mul_overflow_generic_impl<_Tp>(__lhs, __rhs);
    }
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_HOST_ARCH(X86_64)
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
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_HOST_ARCH(X86_64)
    {
      return ::cuda::__mul_overflow_generic_impl<_Tp>(__lhs, __rhs);
    }
  } // ^^^ unsigned types ^^^
}

#endif // _CCCL_HOST_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __mul_overflow_uniform_type(_Tp __lhs, _Tp __rhs) noexcept
{
  // TODO: add_overflow utilizes !_CCCL_TILE_COMPILATION(); is this needed?
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (return ::cuda::__mul_overflow_device(__lhs, __rhs);),
                 (return ::cuda::__mul_overflow_host(__lhs, __rhs);))
  }
  return ::cuda::__mul_overflow_generic_impl(__lhs, __rhs);
}

template <typename _Result, typename _Lhs, typename _Rhs>
inline constexpr bool __is_mul_representable_v =
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
_CCCL_API constexpr overflow_result<_ActualResult> mul_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
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
    if constexpr (sizeof(_ActualResult) != 16 && sizeof(_Lhs) != 16 && sizeof(_Rhs) != 16)
#  endif // _CCCL_COMPILER(NVHPC)
    {
      NV_IF_TARGET(NV_IS_HOST, ({
                     overflow_result<_ActualResult> __result{};
                     __result.overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__lhs, __rhs, &__result.value);
                     return __result;
                   }))
    }
  }
#endif // _CCCL_BUILTIN_MUL_OVERFLOW

  // Host fallback + device implementation.
#if _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_MUL_OVERFLOW) || (_CCCL_COMPILER(NVHPC) && _CCCL_HAS_INT128())
  using ::cuda::std::__num_bits_v;
  using ::cuda::std::is_signed_v;
  using ::cuda::std::is_unsigned_v;
  using _CommonAll                             = ::cuda::std::common_type_t<_Common, _ActualResult>;
  [[maybe_unused]] const bool __is_lhs_ge_zero = is_unsigned_v<_Lhs> || __lhs >= 0;
  [[maybe_unused]] const bool __is_rhs_ge_zero = is_unsigned_v<_Rhs> || __rhs >= 0;
  // shortcut for the case where inputs are representable with the max type
  if constexpr (__is_mul_representable_v<_ActualResult, _Lhs, _Rhs>)
  {
    const auto __lhs1    = static_cast<_CommonAll>(__lhs);
    const auto __rhs1    = static_cast<_CommonAll>(__rhs);
    const auto __product = static_cast<_CommonAll>(__lhs1 * __rhs1);
    return ::cuda::overflow_cast<_ActualResult>(__product);
  }
  // * int x int -> int
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs> && is_signed_v<_ActualResult>) // all signed
  {
    using _Sp            = ::cuda::std::make_signed_t<_CommonAll>;
    const auto __lhs1    = static_cast<_Sp>(__lhs);
    const auto __rhs1    = static_cast<_Sp>(__rhs);
    const auto __product = ::cuda::__mul_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret     = ::cuda::overflow_cast<_ActualResult>(__product.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __product.overflow};
  }
  // Positive inputs
  // * unsigned + unsigned (compile-time)
  // * unsigned + int >= 0 (compile-time + run-time check)
  // * int >= 0 + unsigned (compile-time + run-time check)
  // * int >= 0 + int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (__is_lhs_ge_zero && __is_rhs_ge_zero)
  {
    const auto __lhs1    = static_cast<_CommonAll>(__lhs);
    const auto __rhs1    = static_cast<_CommonAll>(__rhs);
    const auto __product = ::cuda::__mul_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret     = ::cuda::overflow_cast<_ActualResult>(__product.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __product.overflow};
  }
  // Negative inputs
  // * int < 0 + int < 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (!__is_lhs_ge_zero && !__is_rhs_ge_zero)
  {
    using _Up            = ::cuda::std::make_unsigned_t<_CommonAll>;
    const auto __lhs1    = static_cast<_Up>(__lhs);
    const auto __rhs1    = static_cast<_Up>(__rhs);
    const auto __product = __lhs1 * __rhs1;
    const auto __ret     = ::cuda::overflow_cast<_ActualResult>(__product);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __product < 0};
  }
  // Opposite signs
  // * int < 0 + int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above)
  // * int >= 0 + int < 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above)
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs>)
  {
    return ::cuda::overflow_cast<_ActualResult>(static_cast<_Common>(__lhs) * static_cast<_Common>(__rhs));
  }
  // Opposite signs
  // * unsigned + int < 0
  // * int < 0 + unsigned
  else
  {
    // skip checks in cmp_less, cmp_greater, uabs
    if constexpr (is_unsigned_v<_Lhs> && is_signed_v<_Rhs>)
    {
      _CCCL_ASSUME(__rhs < 0);
    }
    else if constexpr (is_signed_v<_Lhs> && is_unsigned_v<_Rhs>)
    {
      _CCCL_ASSUME(__lhs < 0);
    }
    const auto __lhs1       = _CommonAll{::cuda::uabs(__lhs)};
    const auto __rhs1       = _CommonAll{::cuda::uabs(__rhs)};
    const auto __product_lo = __lhs1 * __rhs1;
    const auto __product_hi = ::cuda::mul_hi(__lhs1, __rhs1);

    if constexpr (is_unsigned_v<_ActualResult>)
    {
      return overflow_result<_ActualResult>{static_cast<_ActualResult>(::cuda::neg(__product_lo)), true};
    }
    else
    {
      using _Up                = ::cuda::std::make_unsigned_t<_CommonAll>;
      constexpr auto __min     = ::cuda::std::numeric_limits<_ActualResult>::min();
      const auto __product_max = _Up{::cuda::uabs(__min)};
      const auto __overflow    = __product_hi != 0 || __product_lo > __product_max;
      return overflow_result<_ActualResult>{static_cast<_ActualResult>(::cuda::neg(__product_lo)), __overflow};
    }
  }
#endif // _CCCL_CUDA_COMPILATION() || !_CCCL_BUILTIN_MUL_OVERFLOW || (_CCCL_COMPILER(NVHPC) && _CCCL_HAS_INT128())
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
