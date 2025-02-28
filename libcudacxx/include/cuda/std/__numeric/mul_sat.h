// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___NUMERIC_MUL_SAT_H
#define _LIBCUDACXX___NUMERIC_MUL_SAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_arithmetic_integral.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class __mul_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp
  __clamp_overflow(_Tp __x, _Tp __y, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      if constexpr (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        _LIBCUDACXX_UNUSED_VAR(__x);
        _LIBCUDACXX_UNUSED_VAR(__y);
        __result = _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
      else
      {
        __result = (__x > _Tp{0} && __y > _Tp{0}) || (__x < _Tp{0} && __y < _Tp{0})
                   ? _CUDA_VSTD::numeric_limits<_Tp>::max()
                   : _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
    }
    return __result;
  }

public:
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    if (__x == _Tp{0} || __y == _Tp{0})
    {
      return _Tp{0};
    }

    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if (__x == _Tp{-1})
      {
        if (__y == _CUDA_VSTD::numeric_limits<_Tp>::min())
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::max();
        }
        return static_cast<_Tp>(-__y);
      }

      if (__y == _Tp{-1})
      {
        if (__x == _CUDA_VSTD::numeric_limits<_Tp>::min())
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::max();
        }
        return static_cast<_Tp>(-__x);
      }

      if (__x > _Tp{0} && __y > _Tp{0})
      {
        if (__x > static_cast<_Tp>(_CUDA_VSTD::numeric_limits<_Tp>::max() / __y))
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::max();
        }
      }
      else if (__x < _Tp{0} && __y < _Tp{0})
      {
        if (__x < static_cast<_Tp>(_CUDA_VSTD::numeric_limits<_Tp>::max() / __y))
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::max();
        }
      }
      else if (__x < _Tp{0} && __y > _Tp{0})
      {
        if (__x < static_cast<_Tp>(_CUDA_VSTD::numeric_limits<_Tp>::min() / __y))
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::min();
        }
      }
      else
      {
        if (__x > static_cast<_Tp>(_CUDA_VSTD::numeric_limits<_Tp>::min() / __y))
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::min();
        }
      }
    }
    else
    {
      if (__x > static_cast<_Tp>(_CUDA_VSTD::numeric_limits<_Tp>::max() / __y))
      {
        return _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
    }
    return static_cast<_Tp>(__x * __y);
  }

#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__x, __y, &__result);

    return __clamp_overflow<_Tp>(__x, __y, __result, __overflow);
  }
#endif // _CCCL_BUILTIN_MUL_OVERFLOW
#if !_CCCL_COMPILER(NVRTC)
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if ((__x == _Tp{-1} && __y == _CUDA_VSTD::numeric_limits<_Tp>::min())
          || (__y == _Tp{-1} && __x == _CUDA_VSTD::numeric_limits<_Tp>::min()))
      {
        return _CUDA_VSTD::numeric_limits<_Tp>::max();
      }

#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
      if constexpr (sizeof(_Tp) == 1)
      {
        int16_t __result;
        bool __overflow = _mul_full_overflow_i8(__x, __y, &__result);
        return __clamp_overflow<int8_t>(__x, __y, static_cast<int8_t>(__result), __overflow);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        int16_t __result;
        bool __overflow = _mul_overflow_i16(__x, __y, &__result);
        return __clamp_overflow<int16_t>(__x, __y, __result, __overflow);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        int32_t __result;
        bool __overflow = _mul_overflow_i32(__x, __y, &__result);
        return __clamp_overflow<int32_t>(__x, __y, __result, __overflow);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        int64_t __result;
        bool __overflow = _mul_overflow_i64(__x, __y, &__result);
        return __clamp_overflow<int64_t>(__x, __y, __result, __overflow);
      }
      else
      {
        return __impl_generic(__x, __y);
      }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
      if constexpr (sizeof(_Tp) == 4)
      {
        const int64_t __result = __emul(__x, __y);
        return __clamp_overflow<int32_t>(
          __x,
          __y,
          static_cast<int32_t>(__result),
          __result > _CUDA_VSTD::numeric_limits<int32_t>::max()
            || __result < _CUDA_VSTD::numeric_limits<int32_t>::min());
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        int64_t __hi = __mulh(__x, __y);
        int64_t __lo = __x * __y;
        return __clamp_overflow<int64_t>(__x, __y, __lo, __hi != _Tp{0} && __hi != _Tp{-1});
      }
      else
      {
        return __impl_generic(__x, __y);
      }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
      return __impl_generic(__x, __y);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
    }
    else
    {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
      if constexpr (sizeof(_Tp) == 1)
      {
        uint16_t __result;
        bool __overflow = _mul_full_overflow_u8(__x, __y, &__result);
        return __clamp_overflow<uint8_t>(__x, __y, static_cast<uint8_t>(__result), __overflow);
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        uint16_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u16(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint16_t>(__x, __y, __lo, __overflow);
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        uint32_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u32(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint32_t>(__x, __y, __lo, __overflow);
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        uint64_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u64(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint64_t>(__x, __y, __lo, __overflow);
      }
      else
      {
        return __impl_generic(__x, __y);
      }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
      if constexpr (sizeof(_Tp) == 4)
      {
        const uint64_t __result = __emulu(__x, __y);
        return __clamp_overflow<uint32_t>(
          __x, __y, static_cast<uint32_t>(__result), __result > _CUDA_VSTD::numeric_limits<uint32_t>::max());
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        return __clamp_overflow<uint64_t>(__x, __y, __x * __y, __umulh(__x, __y) != _Tp{0});
      }
      else
      {
        return __impl_generic(__x, __y);
      }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
      return __impl_generic(__x, __y);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
    }
  }
#endif // !_CCCL_COMPILER(NVRTC)
#if _CCCL_HAS_CUDA_COMPILER
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        int16_t __result;
        asm("mul.lo.s16 %0, %1, %2;" : "=h"(__result) : "h"(static_cast<int16_t>(__x)), "h"(static_cast<int16_t>(__y)));
        return static_cast<int8_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::min()),
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        int32_t __result;
        asm("mul.wide.s16 %0, %1, %2;" : "=r"(__result) : "h"(__x), "h"(__y));
        return static_cast<int16_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::min()),
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        int64_t __result;
        asm("mul.wide.s32 %0, %1, %2;" : "=l"(__result) : "r"(__x), "r"(__y));
        return static_cast<int32_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int64_t>(_CUDA_VSTD::numeric_limits<int32_t>::min()),
          static_cast<int64_t>(_CUDA_VSTD::numeric_limits<int32_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        if ((__x == _Tp{-1} && __y == _CUDA_VSTD::numeric_limits<_Tp>::min())
            || (__y == _Tp{-1} && __x == _CUDA_VSTD::numeric_limits<_Tp>::min()))
        {
          return _CUDA_VSTD::numeric_limits<_Tp>::max();
        }

        int64_t __lo, __hi;
        __lo = __x * __y;
        __hi = __mul64hi(__x, __y);
        return __clamp_overflow<int64_t>(__x, __y, __lo, __hi != _Tp{0} && __hi != _Tp{-1});
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
    else
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        uint16_t __result;
        asm("mul.lo.u16 %0, %1, %2;"
            : "=h"(__result)
            : "h"(static_cast<uint16_t>(__x)), "h"(static_cast<uint16_t>(__y)));
        return static_cast<uint8_t>(
          _CUDA_VSTD::min<uint16_t>(__result, static_cast<uint16_t>(_CUDA_VSTD::numeric_limits<uint8_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        uint32_t __result;
        asm("mul.wide.u16 %0, %1, %2;" : "=r"(__result) : "h"(__x), "h"(__y));
        return static_cast<uint16_t>(
          _CUDA_VSTD::min<uint32_t>(__result, static_cast<uint32_t>(_CUDA_VSTD::numeric_limits<uint16_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        uint64_t __result;
        asm("mul.wide.u32 %0, %1, %2;" : "=l"(__result) : "r"(__x), "r"(__y));
        return static_cast<uint32_t>(
          _CUDA_VSTD::min<uint64_t>(__result, static_cast<uint64_t>(_CUDA_VSTD::numeric_limits<uint32_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        uint64_t __lo, __hi;
        __lo = __x * __y;
        __hi = __umul64hi(__x, __y);
        return (__hi == uint64_t{0}) ? __lo : _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
  }
#endif // _CCCL_HAS_CUDA_COMPILER
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp mul_sat(_Tp __x, _Tp __y) noexcept
{
  using _Up = remove_cv_t<_Tp>;
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  return __mul_sat::__impl_builtin<_Up>(__x, __y);
#else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __mul_sat::__impl_host<_Up>(__x, __y);), (return __mul_sat::__impl_device<_Up>(__x, __y);))
  }
  return __mul_sat::__impl_generic<_Up>(__x, __y);
#endif // !_CCCL_BUILTIN_MUL_OVERFLOW
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NUMERIC_MUL_SAT_H
