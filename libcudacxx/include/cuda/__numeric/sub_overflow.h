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

#include <cuda/std/__limits/numeric_limits.h>

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
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
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

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr overflow_result<_Tp> __sub_overflow_generic_impl(_Tp __lhs, _Tp __rhs) noexcept
{
  using _Up  = ::cuda::std::make_unsigned_t<_Tp>;
  auto __sub = static_cast<_Tp>(static_cast<_Up>(__lhs) - static_cast<_Up>(__rhs));
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
        __result.overflow = ::_sub_overflow_i8(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        __result.overflow = ::_sub_overflow_i16(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        __result.overflow = ::_sub_overflow_i32(0, __lhs, __rhs, &__result.value);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        __result.overflow = ::_sub_overflow_i64(0, __lhs, __rhs, &__result.value);
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
          __result.overflow = ::_subborrow_u8(0, __lhs, __rhs, &__result.value);
        }
        else if constexpr (sizeof(_Tp) == 2)
        {
          __result.overflow = ::_subborrow_u16(0, __lhs, __rhs, &__result.value);
        }
        else if constexpr (sizeof(_Tp) == 4)
        {
          __result.overflow = ::_subborrow_u32(0, __lhs, __rhs, &__result.value);
        }
        else if constexpr (sizeof(_Tp) == 8)
        {
          __result.overflow = ::_subborrow_u64(0, __lhs, __rhs, &__result.value);
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
  if (!::cuda::std::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (return ::cuda::__sub_overflow_device(__lhs, __rhs);),
                 (return ::cuda::__sub_overflow_host(__lhs, __rhs);))
  }
  return ::cuda::__sub_overflow_generic_impl(__lhs, __rhs);
}

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
[[nodiscard]] _CCCL_API constexpr overflow_result<_ActualResult>
sub_overflow(const _Lhs __lhs, const _Rhs __rhs) noexcept
{
// (1) __builtin_sub_overflow is not available in a constant expression with gcc + nvcc
// (2) __builtin_sub_overflow generates suboptimal code with nvc++ and clang-cuda for device code
#if defined(_CCCL_BUILTIN_SUB_OVERFLOW) && _CCCL_HOST_COMPILATION() \
  && !(_CCCL_COMPILER(GCC) && _CCCL_CUDA_COMPILER(NVCC))
  overflow_result<_ActualResult> __result;
  __result.overflow = _CCCL_BUILTIN_SUB_OVERFLOW(__lhs, __rhs, &__result.value);
  return __result;
#else
  using ::cuda::std::__make_nbit_int_t;
  using ::cuda::std::__make_nbit_uint_t;
  using ::cuda::std::__num_bits_v;
  using ::cuda::std::is_same_v;
  using ::cuda::std::is_signed_v;
  using ::cuda::std::is_unsigned_v;
  using _CommonAll                             = ::cuda::std::common_type_t<_Common, _ActualResult>;
  [[maybe_unused]] const bool __is_lhs_ge_zero = is_unsigned_v<_Lhs> || __lhs >= 0;
  [[maybe_unused]] const bool __is_rhs_ge_zero = is_unsigned_v<_Rhs> || __rhs >= 0;
  // shortcut for the case where inputs are representable with the common type
  if constexpr (__is_integer_representable_v<_Lhs, _CommonAll> && __is_integer_representable_v<_Rhs, _CommonAll>)
  {
    const auto __lhs1 = static_cast<_CommonAll>(__lhs);
    const auto __rhs1 = static_cast<_CommonAll>(__rhs);
    const auto __sub  = static_cast<_CommonAll>(__lhs1 - __rhs1);
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__sub), false};
  }
  // * signed - signed -> signed
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs> && is_signed_v<_ActualResult>) // all signed
  {
    using _Sp         = __make_nbit_int_t<__num_bits_v<_CommonAll>>;
    const auto __lhs1 = static_cast<_Sp>(__lhs);
    const auto __rhs1 = static_cast<_Sp>(__rhs);
    const auto __sub  = ::cuda::__sub_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret  = ::cuda::overflow_cast<_ActualResult>(__sub.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __sub.overflow};
  }
  // Positive inputs -> handle them as unsigned
  // * unsigned - unsigned (compile-time)
  // * unsigned - int >= 0 (compile-time + run-time check)
  // * int >= 0 - unsigned (compile-time + run-time check)
  // * int >= 0 - int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (__is_lhs_ge_zero && __is_rhs_ge_zero)
  {
    using _Up         = __make_nbit_uint_t<__num_bits_v<_CommonAll>>;
    const auto __lhs1 = static_cast<_Up>(__lhs);
    const auto __rhs1 = static_cast<_Up>(__rhs);
    const auto __sub  = ::cuda::__sub_overflow_uniform_type(__lhs1, __rhs1);
    const auto __ret  = ::cuda::overflow_cast<_ActualResult>(__sub.value);
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __sub.overflow};
  }
  // Negative result with unsigned return type
  // * int < 0 - int >= 0 -> _ActualResult=unsigned (_ActualResult=signed already handled above) (run-time check)
  else if (!__is_lhs_ge_zero && __is_rhs_ge_zero)
  {
    const auto __lhs1 = static_cast<_ActualResult>(__lhs);
    const auto __rhs1 = static_cast<_ActualResult>(__rhs);
    return overflow_result<_ActualResult>{static_cast<_ActualResult>(__lhs1 - __rhs1), true};
  }
  // The result falls into _Common range (opposite signs)
  // * int < 0 - int < 0   -> _ActualResult=unsigned (_ActualResult=signed already handled above)
  // * int >= 0 - int >= 0 -> already handled above
  else if constexpr (is_signed_v<_Lhs> && is_signed_v<_Rhs>)
  {
    const auto __lhs1                  = static_cast<_Common>(__lhs);
    const auto __rhs1                  = static_cast<_Common>(__rhs);
    const auto __ret                   = ::cuda::overflow_cast<_ActualResult>(__lhs1 - __rhs1);
    constexpr auto __common_min        = ::cuda::std::numeric_limits<_Common>::min();
    const bool __is_underflow_edgecase = sizeof(_Rhs) == sizeof(_Common) && __rhs1 == __common_min && __lhs1 != __rhs1;
    return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __is_underflow_edgecase};
  }
  // Opposite type signs
  // * unsigned - int < 0
  // * \\\\\ int < 0 - unsigned
  else
  {
    const auto __lhs1 = static_cast<_Common>(__lhs); // _Common is unsigned
    const auto __rhs1 = static_cast<_Common>(__rhs);
    const auto __sub  = static_cast<_Common>(__lhs1 - __rhs1);
    //if constexpr (is_unsigned_v<_Lhs>) // unsigned - int < 0 -> positive value representable with _Common
    //{
      return ::cuda::overflow_cast<_ActualResult>(__sub);
    //}
    //else // int < 0 - unsigned -> negative value
    //{
    //  using _SignedCommon       = ::cuda::std::make_signed_t<_Common>;
    //  const auto __ret          = ::cuda::overflow_cast<_ActualResult>(static_cast<_SignedCommon>(__sub));
    //  const bool __is_underflow = __sub > ::cuda::uabs{cuda::std::numeric_limits<_SignedCommon>::min()};
    //  return overflow_result<_ActualResult>{__ret.value, __ret.overflow || __is_underflow};
    //}
  }
#endif // defined(_CCCL_BUILTIN_SUB_OVERFLOW) && !_CCCL_CUDA_COMPILER(NVCC)
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
