//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_SINCOS_H
#define _CUDA___CMATH_SINCOS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_sincosf) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINCOSF(...) __builtin_sincosf(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_sincosf) || _CCCL_COMPILER(GCC)

#if _CCCL_HAS_BUILTIN(__builtin_sincos) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINCOS(...) __builtin_sincos(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_sincos) || _CCCL_COMPILER(GCC)

#if _CCCL_HAS_BUILTIN(__builtin_sincosl) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_SINCOSL(...) __builtin_sincosl(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_sincosl) || _CCCL_COMPILER(GCC)

// clang-cuda crashes if these builtins are used.
#if _CCCL_CUDA_COMPILER(CLANG)
#  undef _CCCL_BUILTIN_SINCOSF
#  undef _CCCL_BUILTIN_SINCOS
#  undef _CCCL_BUILTIN_SINCOSL
#endif // _CCCL_CUDA_COMPILER(CLANG)

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Type returned by \c cuda::sincos.
template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT sincos_result
{
  _Tp sin; //!< The sin result.
  _Tp cos; //!< The cos result.
};

//! @brief Computes sin and cos operation of a value.
//!
//! @param __v The value.
//!
//! @return The \c cuda::sincos_result with the results of sin and cos operations.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__is_extended_arithmetic_v<_Tp>)
[[nodiscard]] _CCCL_API auto sincos(_Tp __v) noexcept
  -> sincos_result<::cuda::std::conditional_t<::cuda::std::is_integral_v<_Tp>, double, _Tp>>
{
  if constexpr (::cuda::std::is_integral_v<_Tp>)
  {
    return ::cuda::sincos(static_cast<double>(__v));
  }
  else
  {
    [[maybe_unused]] sincos_result<_Tp> __ret{};
#if defined(_CCCL_BUILTIN_SINCOSF)
    if constexpr (::cuda::std::is_same_v<_Tp, float>)
    {
      _CCCL_BUILTIN_SINCOSF(__v, &__ret.sin, &__ret.cos);
      return __ret;
    }
#endif // _CCCL_BUILTIN_SINCOSF
#if defined(_CCCL_BUILTIN_SINCOS)
    if constexpr (::cuda::std::is_same_v<_Tp, double>)
    {
      _CCCL_BUILTIN_SINCOS(__v, &__ret.sin, &__ret.cos);
      return __ret;
    }
#endif // _CCCL_BUILTIN_SINCOS
#if _CCCL_HAS_LONG_DOUBLE() && defined(_CCCL_BUILTIN_SINCOSL)
    if constexpr (::cuda::std::is_same_v<_Tp, long double>)
    {
      _CCCL_BUILTIN_SINCOSL(__v, &__ret.sin, &__ret.cos);
      return __ret;
    }
#endif // _CCCL_HAS_LONG_DOUBLE() && _CCCL_BUILTIN_SINCOSL

    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      if constexpr (::cuda::std::is_same_v<_Tp, float>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (::sincosf(__v, &__ret.sin, &__ret.cos); return __ret;))
      }
      if constexpr (::cuda::std::is_same_v<_Tp, double>)
      {
        NV_IF_TARGET(NV_IS_DEVICE, (::sincos(__v, &__ret.sin, &__ret.cos); return __ret;))
      }
#if _LIBCUDACXX_HAS_NVFP16()
      if constexpr (::cuda::std::is_same_v<_Tp, ::__half>)
      {
        const auto __result_float = ::cuda::sincos(::__half2float(__v));
        return {::__float2half(__result_float.sin), ::__float2half(__result_float.cos)};
      }
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
      if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>)
      {
        const auto __result_float = ::cuda::sincos(::__bfloat162float(__v));
        return {::__float2bfloat16(__result_float.sin), ::__float2bfloat16(__result_float.cos)};
      }
#endif // _LIBCUDACXX_HAS_NVBF16()
    }
    return {::cuda::std::sin(__v), ::cuda::std::cos(__v)};
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___CMATH_SINCOS_H
