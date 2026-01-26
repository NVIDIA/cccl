// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_TRAITS_H
#define _CUDA_STD___CMATH_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__floating_point/cuda_fp_types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

#if !_CCCL_COMPILER(NVRTC)
#  include <math.h>
#endif // !_CCCL_COMPILER(NVRTC)

#include <cuda/std/__cccl/prologue.h>

#if !_CCCL_COMPILER(NVRTC)
// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef isgreater

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isgreater_runtime(_Tp __x, _Tp __y) noexcept
{
  return isgreater(__x, __y);
}
#    pragma push_macro("isgreater")
#    undef isgreater
#    define _CCCL_POP_MACRO_isgreater
#  else // ^^^ Function Macro isgreater ^^^ / vvv No function macro isgreater vvv
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isgreater_runtime(_Tp __x, _Tp __y) noexcept
{
  return ::isgreater(__x, __y);
}
#  endif // No function macro isgreater

// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef isgreaterequal

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isgreaterequal_runtime(_Tp __x, _Tp __y) noexcept
{
  return isgreaterequal(__x, __y);
}
#    pragma push_macro("isgreaterequal")
#    undef isgreaterequal
#    define _CCCL_POP_MACRO_isgreaterequal
#  else // ^^^ Function Macro isgreaterequal ^^^ / vvv No function macro isgreaterequal vvv
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isgreaterequal_runtime(_Tp __x, _Tp __y) noexcept
{
  return ::isgreaterequal(__x, __y);
}
#  endif // No function macro isgreaterequal

// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef isless

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isless_runtime(_Tp __x, _Tp __y) noexcept
{
  return isless(__x, __y);
}
#    pragma push_macro("isless")
#    undef isless
#    define _CCCL_POP_MACRO_isless
#  else // ^^^ Function Macro isless ^^^ / vvv No function macro isless vvv
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_isless_runtime(_Tp __x, _Tp __y) noexcept
{
  return ::isless(__x, __y);
}
#  endif // No function macro isless

// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef islessequal

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_islessequal_runtime(_Tp __x, _Tp __y) noexcept
{
  return islessequal(__x, __y);
}
#    pragma push_macro("islessequal")
#    undef islessequal
#    define _CCCL_POP_MACRO_islessequal
#  else // ^^^ Function Macro islessequal ^^^ / vvv No function macro islessequal vvv
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_islessequal_runtime(_Tp __x, _Tp __y) noexcept
{
  return ::islessequal(__x, __y);
}
#  endif // No function macro islessequal

// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef islessgreater

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_islessgreater_runtime(_Tp __x, _Tp __y) noexcept
{
  return islessgreater(__x, __y);
}
#    pragma push_macro("islessgreater")
#    undef islessgreater
#    define _CCCL_POP_MACRO_islessgreater
#  else // ^^^ Function Macro islessgreater ^^^ / vvv No function macro islessgreater vvv
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API bool __cccl_islessgreater_runtime(_Tp __x, _Tp __y) noexcept
{
  return ::islessgreater(__x, __y);
}
#  endif // No function macro islessgreater

// This function may or may not be implemented as a function macro so we need to handle that case
#  ifdef isunordered
// No fallback implementation
#    pragma push_macro("isunordered")
#    undef isunordered
#    define _CCCL_POP_MACRO_isunordered
#  endif // Function macro isunordered

#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// isgreater

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API bool __device_isgreater(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x > __y;
}

#if _CCCL_CHECK_BUILTIN(builtin_isgreater) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISGREATER(...) __builtin_isgreater(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_isgreater)

#if !_CCCL_COMPILER(NVRTC)
template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_HOST_API bool __host_isgreater(_A1 __x, _A1 __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ISGREATER)
  return _CCCL_BUILTIN_ISGREATER(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ISGREATER ^^^ / vvv !_CCCL_BUILTIN_ISGREATER vvv
  return ::__cccl_isgreater_runtime(__x, __y);
#  endif // !_CCCL_BUILTIN_ISGREATER
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API bool isgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__host_isgreater((type) __x, (type) __y);),
                    (return ::cuda::std::__device_isgreater((type) __x, (type) __y);))
}

// isgreaterequal

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API bool __device_isgreaterequal(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x >= __y;
}

#if _CCCL_CHECK_BUILTIN(builtin_isgreaterequal) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISGREATEREQUAL(...) __builtin_isgreaterequal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_isgreaterequal)

#if !_CCCL_COMPILER(NVRTC)
template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_HOST_API bool __host_isgreaterequal(_A1 __x, _A1 __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ISGREATEREQUAL)
  return _CCCL_BUILTIN_ISGREATEREQUAL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ISGREATEREQUAL ^^^ / vvv !_CCCL_BUILTIN_ISGREATEREQUAL vvv
  return ::__cccl_isgreaterequal_runtime(__x, __y);
#  endif // !_CCCL_BUILTIN_ISGREATEREQUAL
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API bool isgreaterequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__host_isgreaterequal((type) __x, (type) __y);),
                    (return ::cuda::std::__device_isgreaterequal((type) __x, (type) __y);))
}

// isless

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API bool __device_isless(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x < __y;
}

#if _CCCL_CHECK_BUILTIN(builtin_isless) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISLESS(...) __builtin_isless(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_isless)

#if !_CCCL_COMPILER(NVRTC)
template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_HOST_API bool __host_isless(_A1 __x, _A1 __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ISLESS)
  return _CCCL_BUILTIN_ISLESS(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ISLESS ^^^ / vvv !_CCCL_BUILTIN_ISLESS vvv
  return ::__cccl_isless_runtime(__x, __y);
#  endif // !_CCCL_BUILTIN_ISLESS
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API bool isless(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__host_isless((type) __x, (type) __y);),
                    (return ::cuda::std::__device_isless((type) __x, (type) __y);))
}

// islessequal

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API bool __device_islessequal(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x <= __y;
}

#if _CCCL_CHECK_BUILTIN(builtin_islessequal) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISLESSEQUAL(...) __builtin_islessequal(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_islessequal)

#if !_CCCL_COMPILER(NVRTC)
template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_HOST_API bool __host_islessequal(_A1 __x, _A1 __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ISLESSEQUAL)
  return _CCCL_BUILTIN_ISLESSEQUAL(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ISLESSEQUAL ^^^ / vvv !_CCCL_BUILTIN_ISLESSEQUAL vvv
  return ::__cccl_islessequal_runtime(__x, __y);
#  endif // !_CCCL_BUILTIN_ISLESSEQUAL
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API bool islessequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__host_islessequal((type) __x, (type) __y);),
                    (return ::cuda::std::__device_islessequal((type) __x, (type) __y);))
}

// islessgreater

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API bool __device_islessgreater(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x < __y || __x > __y;
}

#if _CCCL_CHECK_BUILTIN(builtin_islessgreater) || _CCCL_COMPILER(GCC)
#  define _CCCL_BUILTIN_ISLESSGREATER(...) __builtin_islessgreater(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_islessgreater)

#if !_CCCL_COMPILER(NVRTC)
template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_HOST_API bool __host_islessgreater(_A1 __x, _A1 __y) noexcept
{
#  if defined(_CCCL_BUILTIN_ISLESSGREATER)
  return _CCCL_BUILTIN_ISLESSGREATER(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ISLESSGREATER ^^^ / vvv !_CCCL_BUILTIN_ISLESSGREATER vvv
  return ::__cccl_islessgreater_runtime(__x, __y);
#  endif // !_CCCL_BUILTIN_ISLESSGREATER
}
#endif // !_CCCL_COMPILER(NVRTC)

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API bool islessgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__host_islessgreater((type) __x, (type) __y);),
                    (return ::cuda::std::__device_islessgreater((type) __x, (type) __y);))
}

// isunordered

template <class _A1, class _A2, enable_if_t<__is_extended_arithmetic_v<_A1> && __is_extended_arithmetic_v<_A2>, int> = 0>
[[nodiscard]] _CCCL_API inline bool isunordered(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  return ::cuda::std::isnan((type) __x) || ::cuda::std::isnan((type) __y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#ifdef _CCCL_POP_MACRO_isgreater
#  pragma pop_macro("isgreater")
#  undef _CCCL_POP_MACRO_isgreater
#endif

#ifdef _CCCL_POP_MACRO_isgreaterequal
#  pragma pop_macro("isgreaterequal")
#  undef _CCCL_POP_MACRO_isgreaterequal
#endif

#ifdef _CCCL_POP_MACRO_isless
#  pragma pop_macro("isless")
#  undef _CCCL_POP_MACRO_isless
#endif

#ifdef _CCCL_POP_MACRO_islessequal
#  pragma pop_macro("islessequal")
#  undef _CCCL_POP_MACRO_islessequal
#endif

#ifdef _CCCL_POP_MACRO_islessgreater
#  pragma pop_macro("islessgreater")
#  undef _CCCL_POP_MACRO_islessgreater
#endif

#ifdef _CCCL_POP_MACRO_isunordered
#  pragma pop_macro("isunordered")
#  undef _CCCL_POP_MACRO_isunordered
#endif

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_TRAITS_H
