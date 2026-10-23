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
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__host_stdlib/math.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/promote.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// maps the bits of a double to an int64_t with the same order
// -0 and +0 map to 0, NaNs have to handled by the caller
[[nodiscard]] _CCCL_API inline int64_t __fp64_to_ordered_int(double __x) noexcept
{
  const auto __x_as_int64 = static_cast<int64_t>(::cuda::std::__fp_get_storage(__x));
  return __x_as_int64 < 0 ? ::cuda::std::numeric_limits<int64_t>::min() - __x_as_int64 : __x_as_int64;
}

// isgreater

template <class _A1, enable_if_t<__is_extended_arithmetic_v<_A1>, int> = 0>
[[nodiscard]] _CCCL_API bool __device_isgreater(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x > __y;
}

[[nodiscard]] _CCCL_API inline bool __device_isgreater(double __x, double __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  _CCCL_DISPATCH_SLOW_FP64_TARGET(
    return ::cuda::std::__fp64_to_ordered_int(__x) > ::cuda::std::__fp64_to_ordered_int(__y);)
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
  return ::isgreater(__x, __y);
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
[[nodiscard]] _CCCL_API bool __device_isgreaterequal(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x >= __y;
}

[[nodiscard]] _CCCL_API inline bool __device_isgreaterequal(double __x, double __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  _CCCL_DISPATCH_SLOW_FP64_TARGET(
    return ::cuda::std::__fp64_to_ordered_int(__x) >= ::cuda::std::__fp64_to_ordered_int(__y);)
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
  return ::isgreaterequal(__x, __y);
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
[[nodiscard]] _CCCL_API bool __device_isless(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x < __y;
}

[[nodiscard]] _CCCL_API inline bool __device_isless(double __x, double __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  _CCCL_DISPATCH_SLOW_FP64_TARGET(
    return ::cuda::std::__fp64_to_ordered_int(__x) < ::cuda::std::__fp64_to_ordered_int(__y);)
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
  return ::isless(__x, __y);
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
[[nodiscard]] _CCCL_API bool __device_islessequal(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x <= __y;
}

[[nodiscard]] _CCCL_API inline bool __device_islessequal(double __x, double __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  _CCCL_DISPATCH_SLOW_FP64_TARGET(
    return ::cuda::std::__fp64_to_ordered_int(__x) <= ::cuda::std::__fp64_to_ordered_int(__y);)
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
  return ::islessequal(__x, __y);
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
[[nodiscard]] _CCCL_API bool __device_islessgreater(_A1 __x, _A1 __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  return __x < __y || __x > __y;
}

[[nodiscard]] _CCCL_API inline bool __device_islessgreater(double __x, double __y) noexcept
{
  if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
  {
    return false;
  }
  // Non-NaN values are unequal iff their bits differ, except for -0 and +0
  // Comparing the bits directly is cheaper than mapping both values to ordered integers
  // ((x | y) << 1) == 0 drops the sign bit and is true only if both are +-0
  _CCCL_DISPATCH_SLOW_FP64_TARGET(
    const auto __x_bits = ::cuda::std::__fp_get_storage(__x); const auto __y_bits = ::cuda::std::__fp_get_storage(__y);
    return __x_bits != __y_bits && ((__x_bits | __y_bits) << 1) != 0;)
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
  return ::islessgreater(__x, __y);
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

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_TRAITS_H
