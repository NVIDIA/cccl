// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_TRAITS_H
#define _LIBCUDACXX___CMATH_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/common.h>
#include <cuda/std/__cmath/fpclassify.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBIT
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#endif // !_CCCL_BUILTIN_SIGNBIT
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_SIGNBIT)
  return _CCCL_BUILTIN_SIGNBIT(__x);
#  else // ^^^ _CCCL_BUILTIN_SIGNBIT ^^^ / vvv !_CCCL_BUILTIN_SIGNBIT vvv
  return ::signbit(__x);
#  endif // !_CCCL_BUILTIN_SIGNBIT
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(__half __x) noexcept
{
  return _CUDA_VSTD::signbit(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::signbit(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1) && _CCCL_TRAIT(is_signed, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(_A1 __x) noexcept
{
  return __x < 0;
}

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1) && !_CCCL_TRAIT(is_signed, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool signbit(_A1) noexcept
{
  return false;
}

// isfinite

#if defined(_CCCL_BUILTIN_ISFINITE) || (defined(_CCCL_BUILTIN_ISINF) && defined(_CCCL_BUILTIN_ISNAN))
#  define _CCCL_CONSTEXPR_ISFINITE constexpr
#else // ^^^ _CCCL_BUILTIN_ISFINITE ^^^ / vvv !_CCCL_BUILTIN_ISFINITE vvv
#  define _CCCL_CONSTEXPR_ISFINITE
#endif // !_CCCL_BUILTIN_ISFINITE

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isfinite(_A1) noexcept
{
  return true;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return !::__isinf(__x) && !::__isnan(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isfinite(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return !::__isinf(__x) && !::__isnan(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isfinite(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISFINITE bool isfinite(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISFINITE)
  return _CCCL_BUILTIN_ISFINITE(__x);
#  elif _CCCL_CUDACC_BELOW(11, 8)
  return !::__isinf(__x) && !::__isnan(__x);
#  else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isfinite(__x);
#  endif // !_CCCL_CUDACC_BELOW(11, 8)
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isfinite(__half __x) noexcept
{
  return !::__hisnan(__x) && !::__hisinf(__x);
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isfinite(__nv_bfloat16 __x) noexcept
{
  return !::__hisnan(__x) && !::__hisinf(__x);
}
#endif // _LIBCUDACXX_HAS_NVBF16

// isinf

#if defined(_CCCL_BUILTIN_ISINF)
#  define _CCCL_CONSTEXPR_ISINF constexpr
#else // ^^^ _CCCL_BUILTIN_ISINF ^^^ / vvv !_CCCL_BUILTIN_ISINF vvv
#  define _CCCL_CONSTEXPR_ISINF
#endif // !_CCCL_BUILTIN_ISINF

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isinf(_A1) noexcept
{
  return false;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISINF bool isinf(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISINF)
  return _CCCL_BUILTIN_ISINF(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isinf(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isinf(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISINF bool isinf(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISINF)
  return _CCCL_BUILTIN_ISINF(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isinf(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isinf(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISINF bool isinf(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISINF)
  return _CCCL_BUILTIN_ISINF(__x);
#  elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isinf(__x);
#  else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isinf(__x);
#  endif // !_CCCL_CUDACC_BELOW(11, 8)
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isinf(__half __x) noexcept
{
#  if _CCCL_STD_VER >= 2020 && _CCCL_CUDACC_BELOW(12, 3)
  // this is a workaround for nvbug 4362808
  return !::__hisnan(__x) && ::__hisnan(__x - __x);
#  else // ^^^ C++20 && below 12.3 ^^^ / vvv C++17 or 12.3+ vvv
  return ::__hisinf(__x) != 0;
#  endif // _CCCL_STD_VER <= 2017 || _CCCL_CUDACC_BELOW(12, 3)
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isinf(__nv_bfloat16 __x) noexcept
{
#  if _CCCL_STD_VER >= 2020 && _CCCL_CUDACC_BELOW(12, 3)
  // this is a workaround for nvbug 4362808
  return !::__hisnan(__x) && ::__hisnan(__x - __x);
#  else // ^^^ C++20 && below 12.3 ^^^ / vvv C++17 or 12.3+ vvv
  return ::__hisinf(__x) != 0;
#  endif // _CCCL_STD_VER <= 2017 || _CCCL_CUDACC_BELOW(12, 3)
}
#endif // _LIBCUDACXX_HAS_NVBF16

// isnan

#if defined(_CCCL_BUILTIN_ISNAN)
#  define _CCCL_CONSTEXPR_ISNAN constexpr
#else // ^^^ _CCCL_BUILTIN_ISNAN ^^^ / vvv !_CCCL_BUILTIN_ISNAN vvv
#  define _CCCL_CONSTEXPR_ISNAN
#endif // !_CCCL_BUILTIN_ISNAN

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool isnan(_A1) noexcept
{
  return false;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(float __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isnan(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isnan(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(double __x) noexcept
{
#if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isnan(__x);
#else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isnan(__x);
#endif // !_CCCL_CUDACC_BELOW(11, 8)
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_ISNAN bool isnan(long double __x) noexcept
{
#  if defined(_CCCL_BUILTIN_ISNAN)
  return _CCCL_BUILTIN_ISNAN(__x);
#  elif _CCCL_CUDACC_BELOW(11, 8)
  return ::__isnan(__x);
#  else // ^^^ _CCCL_CUDACC_BELOW(11, 8) ^^^ / vvv !_CCCL_CUDACC_BELOW(11, 8) vvv
  return ::isnan(__x);
#  endif // !_CCCL_CUDACC_BELOW(11, 8)
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnan(__half __x) noexcept
{
  return ::__hisnan(__x);
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnan(__nv_bfloat16 __x) noexcept
{
  return ::__hisnan(__x);
}
#endif // _LIBCUDACXX_HAS_NVBF16

// isnormal

template <class _A1, enable_if_t<_CCCL_TRAIT(is_integral, _A1), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(_A1 __x) noexcept
{
  return __x != 0;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(float __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(double __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}

#if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(long double __x) noexcept
{
  return _CUDA_VSTD::fpclassify(__x) == FP_NORMAL;
}
#endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

#if defined(_LIBCUDACXX_HAS_NVFP16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(__half __x) noexcept
{
  return _CUDA_VSTD::isnormal(__half2float(__x));
}
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isnormal(__nv_bfloat16 __x) noexcept
{
  return _CUDA_VSTD::isnormal(__bfloat162float(__x));
}
#endif // _LIBCUDACXX_HAS_NVBF16

// isgreater

template <class _Tp>
struct __is_extended_arithmetic
{
  static constexpr bool value = _CCCL_TRAIT(is_arithmetic, _Tp) || _CCCL_TRAIT(__is_extended_floating_point, _Tp);
};

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_CCCL_INLINE_VAR constexpr bool __is_extended_arithmetic_v =
  is_arithmetic_v<_Tp> || __is_extended_floating_point_v<_Tp>;
#endif // !_CCCL_NO_INLINE_VARIABLES

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isgreater(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x > __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isgreater((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isgreater((type) __x, (type) __y);))
}

// isgreaterequal

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isgreaterequal(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x >= __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isgreaterequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isgreaterequal((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isgreaterequal((type) __x, (type) __y);))
}

// isless

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_isless(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x < __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isless(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::isless((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_isless((type) __x, (type) __y);))
}

// islessequal

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_islessequal(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x <= __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool islessequal(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::islessequal((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_islessequal((type) __x, (type) __y);))
}

// islessgreater

template <class _A1, enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1), int> = 0>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI bool __device_islessgreater(_A1 __x, _A1 __y) noexcept
{
  if (_CUDA_VSTD::isnan(__x) || _CUDA_VSTD::isnan(__y))
  {
    return false;
  }
  return __x < __y || __x > __y;
}

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool islessgreater(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::islessgreater((type) __x, (type) __y);),
                    (return _CUDA_VSTD::__device_islessgreater((type) __x, (type) __y);))
}

// isunordered

template <class _A1,
          class _A2,
          enable_if_t<_CCCL_TRAIT(__is_extended_arithmetic, _A1) && _CCCL_TRAIT(__is_extended_arithmetic, _A2), int> = 0>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI bool isunordered(_A1 __x, _A2 __y) noexcept
{
  using type = __promote_t<_A1, _A2>;
  return _CUDA_VSTD::isnan((type) __x) || _CUDA_VSTD::isnan((type) __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CMATH_TRAITS_H
