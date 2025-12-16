//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___NUMERIC_SUB_OVERFLOW_H
#define _CUDA___NUMERIC_SUB_OVERFLOW_H

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
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <nv/target>

#if _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_sub_overflow) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SUB_OVERFLOW(...) __builtin_sub_overflow(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_sub_overflow)

// nvc++ doesn't support 128-bit integers and crashes when certain type combinations are used (nvbug 5730860), so let's
// just disable the builtin for now.
#if _CCCL_COMPILER(NVHPC)
#  undef _CCCL_BUILTIN_SUB_OVERFLOW
#endif // _CCCL_COMPILER(NVHPC)

_CCCL_BEGIN_NAMESPACE_CUDA

// Subtracts using unsigned intermediates to avoid signed overflow (UB) before casting to the requested result type.
template <typename _Result, typename _Lhs, typename _Rhs>
[[nodiscard]] _CCCL_API constexpr _Result __sub_as_unsigned(_Lhs __lhs, _Rhs __rhs) noexcept
{
  using _UnsignedResult = ::cuda::std::make_unsigned_t<_Result>;
  const auto __lhs1     = static_cast<_UnsignedResult>(__lhs);
  const auto __rhs1     = static_cast<_UnsignedResult>(__rhs);
  return static_cast<_Result>(__lhs1 - __rhs1);
}

// addition with unsigned types to avoid UB, return an unsigned type
template <typename _Result, typename _Lhs, typename _Rhs>
[[nodiscard]] _CCCL_API constexpr _Result __add_as_unsigned(_Lhs __lhs, _Rhs __rhs) noexcept
{
  using _UnsignedResult = ::cuda::std::make_unsigned_t<_Result>;
  const auto __lhs1     = static_cast<_UnsignedResult>(__lhs);
  const auto __rhs1     = static_cast<_UnsignedResult>(__rhs);
  return static_cast<_Result>(__lhs1 + __rhs1);
}

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __sub_overflow_generic_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  const auto __sub = ::cuda::__sub_as_unsigned<_Tp>(__lhs, __rhs);
  if constexpr (::cuda::std::is_signed_v<_Tp>)
  {
    return overflow_result<_Tp>{__sub, (__sub > __lhs) == (__rhs >= _Tp{0})};
  }
  else
  {
    return overflow_result<_Tp>{__sub, __sub > __lhs};
  }
}

#if _CCCL_DEVICE_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API overflow_result<_Tp> __sub_overflow_device(_Tp __lhs, _Tp __rhs) noexcept
{
  // signed version of sub.cc behave the same way of unsigned integers
  if constexpr ((sizeof(_Tp) == 4 || sizeof(_Tp) == 8) && ::cuda::std::is_unsigned_v<_Tp>)
  {
    _Tp __result;
    int __overflow;
    if constexpr (sizeof(_Tp) == 4)
    {
      asm("sub.cc.u32 %0, %2, %3;"
          "subc.u32 %1, 0, 0;"
          : "=r"(__result), "=r"(__overflow)
          : "r"(__lhs), "r"(__rhs));
    }
    else if constexpr (sizeof(_Tp) == 8)
    {
      asm("sub.cc.u64 %0, %2, %3;"
          "subc.u32 %1, 0, 0;"
          : "=l"(__result), "=r"(__overflow)
          : "l"(__lhs), "l"(__rhs));
    }
    return overflow_result<_Tp>{__result, static_cast<bool>(__overflow)};
  }
  else
  {
    return ::cuda::__sub_overflow_generic_impl(__lhs, __rhs); // do not use builtin functions
  }
}

#endif // _CCCL_DEVICE_COMPILATION()

#if _CCCL_HOST_COMPILATION()

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API overflow_result<_Tp> __sub_overflow_host(_Tp __lhs, _Tp __rhs) noexcept
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
        __result.overflow = ::_sub_overflow_i8(0, __lhs, __rhs, reinterpret_cast<signed char*>(&__result.value));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = ::_sub_overflow_i16(0, __lhs, __rhs, reinterpret_cast<short*>(&__result.value));
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = ::_sub_overflow_i32(0, __lhs, __rhs, reinterpret_cast<int*>(&__result.value));
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = ::_sub_overflow_i64(0, __lhs, __rhs, reinterpret_cast<long long*>(&__result.value));
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
          __result.overflow = ::_subborrow_u8(0, __lhs, __rhs, reinterpret_cast<unsigned char*>(&__result.value));
        }
        else if constexpr (sizeof(_Tp) == 2)
        {
          __result.overflow = ::_subborrow_u16(0, __lhs, __rhs, reinterpret_cast<unsigned short*>(&__result.value));
        }
        else if constexpr (sizeof(_Tp) == 4)
        {
          __result.overflow = ::_subborrow_u32(0, __lhs, __rhs, reinterpret_cast<unsigned int*>(&__result.value));
        }
        else if constexpr (sizeof(_Tp) == 8)
        {
          __result.overflow = ::_subborrow_u64(0, __lhs, __rhs, reinterpret_cast<unsigned long long*>(&__result.value));
        }
        return __result;
      }
      else
      {
        return ::cuda::__sub_overflow_generic_impl(__lhs, __rhs);
      }
  }
  else
#  endif // ^^^ _CCCL_COMPILER(MSVC) || _CCCL_ARCH(X86_64) ^^^
  {
    return ::cuda::__sub_overflow_generic_impl(__lhs, __rhs);
  }
}

#endif // _CCCL_HOST_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __sub_overflow_uniform_type(_Tp __lhs, _Tp __rhs) noexcept
{
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (return ::cuda::__sub_overflow_device(__lhs, __rhs);),
                 (return ::cuda::__sub_overflow_host(__lhs, __rhs);))
  }
  return ::cuda::__sub_overflow_generic_impl(__lhs, __rhs);
}

template <typename _Result, typename _Lhs, typename _Rhs>
inline constexpr bool __is_sub_representable_v = sizeof(_Result) > sizeof(_Lhs) && sizeof(_Result) > sizeof(_Rhs)
                                              && ::cuda::std::is_signed_v<_Result>;

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
_CCCL_API constexpr overflow_result<_ActualResult> sub_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  // We want to use __builtin_sub_overflow only in host code. When compiling CUDA source file, we cannot use it in
  // constant expressions, because it doesn't work before nvcc 13.1 and is buggy in 13.1. When compiling C++ source
  // file, we can use it all the time.
#if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
#  if _CCCL_CUDA_COMPILATION()
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
#  endif // _CCCL_CUDA_COMPILATION()
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   overflow_result<_ActualResult> __result{};
                   __result.overflow = _CCCL_BUILTIN_SUB_OVERFLOW(__lhs, __rhs, &__result.value);
                   return __result;
                 }))
  }
#endif // _CCCL_BUILTIN_SUB_OVERFLOW

  // Host fallback + device implementation.
#if _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_SUB_OVERFLOW)
  using ::cuda::std::common_type_t;
  using ::cuda::std::is_signed_v;
  using ::cuda::std::is_unsigned_v;
  using ::cuda::std::make_signed_t;
  using ::cuda::std::make_unsigned_t;
  using ::cuda::std::numeric_limits;
  // shortcut for the case where inputs are representable with the result type
  if constexpr (__is_sub_representable_v<_ActualResult, _Lhs, _Rhs>)
  {
    constexpr auto __max_lhs_rhs = sizeof(_Lhs) > sizeof(_Rhs) ? sizeof(_Lhs) : sizeof(_Rhs);
    constexpr auto __max_bits = __max_lhs_rhs * 2 <= sizeof(_ActualResult) ? __max_lhs_rhs * 2 : sizeof(_ActualResult);
    using _ComputeType        = ::cuda::std::__make_nbit_int_t<__max_bits * 8>;
    const auto __lhs1         = static_cast<_ComputeType>(__lhs);
    const auto __rhs1         = static_cast<_ComputeType>(__rhs);
    const auto __sub          = static_cast<_ComputeType>(__lhs1 - __rhs1);
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__sub), false};
  }
  // all types have the same sign
  else if constexpr (is_signed_v<_Lhs> == is_signed_v<_Rhs> && is_signed_v<_Lhs> == is_signed_v<_ActualResult>)
  {
    using _CommonAll  = common_type_t<_Common, _ActualResult>;
    const auto __lhs1 = static_cast<_CommonAll>(__lhs);
    const auto __rhs1 = static_cast<_CommonAll>(__rhs);
    const auto __sub  = ::cuda::__sub_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret  = ::cuda::overflow_cast<_ActualResult>(__sub.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __sub.overflow};
  }
  else if (::cuda::std::cmp_less(__lhs, __rhs)) // lhs < rhs -> negative result
  {
    if constexpr (is_unsigned_v<_ActualResult>) // if _ActualResult is unsigned, any negative result is an underflow
    {
      const auto __lhs1 = static_cast<_ActualResult>(__lhs);
      const auto __rhs1 = static_cast<_ActualResult>(__rhs);
      return overflow_result<_ActualResult>{static_cast<_ActualResult>(__lhs1 - __rhs1), true};
    }
    else
    {
      // perform the subtraction as signed (negative result) and check if the result is out of range
      // Then, there are two cases depending on the sign of the rhs
      using _SignedCommonAll          = make_signed_t<common_type_t<_Common, _ActualResult>>;
      const auto __sub                = ::cuda::__sub_as_unsigned<_SignedCommonAll>(__lhs, __rhs);
      constexpr auto __result_min     = numeric_limits<_ActualResult>::min();
      const auto __is_out_of_range    = ::cuda::std::cmp_less(__sub, __result_min);
      const auto __sub_ret            = static_cast<_ActualResult>(__sub);
      const bool __rhs_less_than_zero = !is_unsigned_v<_Rhs> && __rhs < _Rhs{0};
      if (__rhs_less_than_zero || __is_out_of_range) // if rhs < 0, lhs - rhs > lhs -> no overflow
      {
        return overflow_result<_ActualResult>{__sub_ret, __is_out_of_range};
      }
      else // rhs >= 0 -> lhs - rhs < result_min? -> lhs < result_min + rhs
      {
        // Now, the problem is to compute 'result_min + rhs' correctly
        //      note: rhs >= 0, result_min < 0
        // * if sizeof(_ActualResult) >= sizeof(_Rhs) or _Rhs is signed, we can use the signed common type
        //   because the sum is always representable
        // * if sizeof(_ActualResult) < sizeof(_Rhs) and _Rhs is unsigned, we can still use the signed common type
        //   if rhs <= result_max because the sum is always representable, e.g. INT_MIN + uint64_t{UINT_MAX} = INT_MAX
        constexpr auto __signed_min = numeric_limits<_ActualResult>::min();
        constexpr auto __result_max = numeric_limits<make_signed_t<_ActualResult>>::max();
        if (sizeof(_ActualResult) >= sizeof(_Rhs)
            || is_signed_v<_Rhs> || ::cuda::std::cmp_less_equal(__rhs, __result_max))
        {
          // use __add_as_unsigned to avoid UB with INT_MIN
          using _SumType            = make_signed_t<common_type_t<_Rhs, _ActualResult>>;
          const auto __usum         = ::cuda::__add_as_unsigned<_SumType>(__signed_min, __rhs);
          const bool __is_underflow = ::cuda::std::cmp_less(__lhs, __usum);
          return overflow_result<_ActualResult>{__sub_ret, __is_underflow};
        }
        else // * otherwise, rhs > result_max and we need to use the unsigned common type
        {
          using _SumType            = make_unsigned_t<common_type_t<_Rhs, _ActualResult>>;
          const auto __usum         = ::cuda::__add_as_unsigned<_SumType>(__signed_min, __rhs);
          const bool __is_underflow = ::cuda::std::cmp_less(__lhs, __usum);
          return overflow_result<_ActualResult>{__sub_ret, __is_underflow};
        }
      }
    }
  }
  else // lhs >= rhs -> positive result
  {
    // perform the subtraction as unsigned (positive result) and check if the result is out of range
    // Then, there are two cases depending on the sign of the rhs
    using _UnsignedCommonAll     = make_unsigned_t<common_type_t<_Common, _ActualResult>>;
    const auto __sub             = ::cuda::__sub_as_unsigned<_UnsignedCommonAll>(__lhs, __rhs);
    const auto __sub_ret         = static_cast<_ActualResult>(__sub);
    constexpr auto __result_max  = numeric_limits<_ActualResult>::max();
    const auto __is_out_of_range = ::cuda::std::cmp_greater(__sub, __result_max);
    const bool __is_rhs_ge_zero  = is_unsigned_v<_Rhs> || __rhs >= 0;
    if (__is_rhs_ge_zero || __is_out_of_range) // rhs >= 0 -> lhs - rhs < lhs -> no overflow
    {
      return overflow_result<_ActualResult>{__sub_ret, __is_out_of_range};
    }
    else // lhs >= 0 && rhs < 0 -> lhs - rhs > result_max? -> lhs > result_max + rhs
    {
      using _Up                     = make_unsigned_t<common_type_t<_Rhs, _ActualResult>>;
      constexpr auto __unsigned_max = numeric_limits<_ActualResult>::max();
      const auto __sum              = ::cuda::__add_as_unsigned<_Up>(__unsigned_max, __rhs);
      const bool __is_overflow      = ::cuda::std::cmp_greater(__lhs, __sum);
      return overflow_result<_ActualResult>{__sub_ret, __is_overflow};
    }
  }
#endif // _CCCL_CUDA_COMPILATION() || !defined(_CCCL_BUILTIN_SUB_OVERFLOW)
}

//! @brief Subtracts two numbers \p __lhs and \p __rhs with overflow detection
_CCCL_TEMPLATE(typename _Result, typename _Lhs, typename _Rhs)
_CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Result> _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Lhs>
                 _CCCL_AND ::cuda::std::__cccl_is_integer_v<_Rhs>)
[[nodiscard]] _CCCL_API constexpr bool sub_overflow(_Result& __result, const _Lhs __lhs, const _Rhs __rhs) noexcept
{
  const auto __res = ::cuda::sub_overflow<_Result>(__lhs, __rhs);
  __result         = __res.value;
  return __res.overflow;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___NUMERIC_SUB_OVERFLOW_H
