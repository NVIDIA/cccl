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

#ifndef _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H
#define _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/std/__algorithm/clamp.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/is_arithmetic_integral.h>
#  include <cuda/std/__type_traits/is_constant_evaluated.h>
#  include <cuda/std/__type_traits/is_signed.h>
#  include <cuda/std/__type_traits/is_unsigned.h>
#  include <cuda/std/__type_traits/make_unsigned.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/climits>
#  include <cuda/std/cstdint>
#  include <cuda/std/limits>

#  include <nv/target>

#  if _CCCL_COMPILER(MSVC)
#    include <intrin.h>
#  endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

class __add_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp
  __clamp_overflow(_Tp, _Tp __y, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      if constexpr (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        _LIBCUDACXX_UNUSED_VAR(__y);
        __result = _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
      else
      {
        __result = (__y < _Tp{0}) ? _CUDA_VSTD::numeric_limits<_Tp>::min() : _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
    }
    return __result;
  }

public:
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      using _Up    = make_unsigned_t<_Tp>;
      _Tp __result = static_cast<_Tp>(static_cast<_Up>(__x) + static_cast<_Up>(__y));
      return __clamp_overflow(__x, __y, __result, (__result < __x) == !(__y < static_cast<_Tp>(0)));
    }
    else
    {
      _Tp __result = static_cast<_Tp>(__x + __y);
      return __clamp_overflow(__x, __y, __result, (__result < __x));
    }
  }

#  if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__x, __y, &__result);

    return __clamp_overflow(__x, __y, __result, __overflow);
  }
#  endif // _CCCL_BUILTIN_ADD_OVERFLOW
#  if !_CCCL_COMPILER(NVRTC)
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_i8(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int8_t __result;
        bool __overflow = _add_overflow_i8(0, __x, __y, &__result);
        return __clamp_overflow<int8_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_i16(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int16_t __result;
        bool __overflow = _add_overflow_i16(0, __x, __y, &__result);
        return __clamp_overflow<int16_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_i32(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int32_t __result;
        bool __overflow = _add_overflow_i32(0, __x, __y, &__result);
        return __clamp_overflow<int32_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_i64(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int64_t __result;
        bool __overflow = _add_overflow_i64(0, __x, __y, &__result);
        return __clamp_overflow<int64_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
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
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_u8(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint8_t __result;
        bool __overflow = _addcarry_u8(0, __x, __y, &__result);
        return __clamp_overflow<uint8_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_u16(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint16_t __result;
        bool __overflow = _addcarry_u16(0, __x, __y, &__result);
        return __clamp_overflow<uint16_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_u32(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint32_t __result;
        bool __overflow = _addcarry_u32(0, __x, __y, &__result);
        return __clamp_overflow<uint32_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_add_u64(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint64_t __result;
        bool __overflow = _addcarry_u64(0, __x, __y, &__result);
        return __clamp_overflow<uint64_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
  }
#  endif // !_CCCL_COMPILER(NVRTC)
#  if _CCCL_HAS_CUDA_COMPILER
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        int16_t __result;
        asm("add.s16 %0, %1, %2;" : "=h"(__result) : "h"(static_cast<int16_t>(__x)), "h"(static_cast<int16_t>(__y)));
        return static_cast<int8_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::min()),
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        int32_t __result = static_cast<int32_t>(__x) + static_cast<int32_t>(__y);
        return static_cast<int16_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::min()),
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        int32_t __result;
        asm("add.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
        return __result;
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        int64_t __result;
        asm("{\n\t"
            ".reg .pred p, q, r;\n\t"
            ".reg .s64 temp;\n\t"
            "add.s64 %0, %1, %2;\n\t"
            "setp.lt.s64 p, %0, %1;\n\t"
            "setp.lt.s64 q, %2, 0;\n\t"
            "xor.pred r, p, q;\n\t"
            "selp.s64 temp, -9223372036854775808, 9223372036854775807, q;\n\t"
            "selp.s64 %0, temp, %0, r;\n\t"
            "}"
            : "=l"(__result)
            : "l"(__x), "l"(__y));
        return __result;
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
    else
    {
      const _Tp __bneg_x = static_cast<_Tp>(~__x);
      return static_cast<_Tp>(__x + _CUDA_VSTD::min<_Tp>(__y, __bneg_x));
    }
  }
#  endif // _CCCL_HAS_CUDA_COMPILER
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp add_sat(_Tp __x, _Tp __y) noexcept
{
  using _Up = remove_cv_t<_Tp>;
#  if defined(_CCCL_BUILTIN_ADD_OVERFLOW)
  return __add_sat::__impl_builtin<_Up>(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_ADD_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_ADD_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __add_sat::__impl_host<_Up>(__x, __y);), (return __add_sat::__impl_device<_Up>(__x, __y);))
  }
  return __add_sat::__impl_generic<_Up>(__x, __y);
#  endif // !_CCCL_BUILTIN_ADD_OVERFLOW
}

class __sub_sat
{
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp
  __clamp_overflow(_Tp, _Tp __y, _Tp __result, bool __overflow) noexcept
  {
    if (__overflow)
    {
      if constexpr (_CCCL_TRAIT(is_unsigned, _Tp))
      {
        _LIBCUDACXX_UNUSED_VAR(__y);
        __result = _CUDA_VSTD::numeric_limits<_Tp>::min();
      }
      else
      {
        __result = (__y > _Tp{0}) ? _CUDA_VSTD::numeric_limits<_Tp>::min() : _CUDA_VSTD::numeric_limits<_Tp>::max();
      }
    }
    return __result;
  }

public:
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_generic(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      using _Up = make_unsigned_t<_Tp>;

      _Tp __result = static_cast<_Tp>(static_cast<_Up>(__x) - static_cast<_Up>(__y));

      return __clamp_overflow(__x, __y, __result, (__result < __x) == !(__y > _Tp{0}));
    }
    else
    {
      _Tp __result = static_cast<_Tp>(__x - __y);

      return __clamp_overflow(__x, __y, __result, (__result > __x));
    }
  }

#  if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_SUB_OVERFLOW(__x, __y, &__result);

    return __clamp_overflow(__x, __y, __result, __overflow);
  }
#  endif // _CCCL_BUILTIN_SUB_OVERFLOW
#  if !_CCCL_COMPILER(NVRTC)
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static _Tp __impl_host(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_i8(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int8_t __result;
        bool __overflow = _sub_overflow_i8(0, __x, __y, &__result);
        return __clamp_overflow<int8_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_i16(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int16_t __result;
        bool __overflow = _sub_overflow_i16(0, __x, __y, &__result);
        return __clamp_overflow<int16_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_i32(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int32_t __result;
        bool __overflow = _sub_overflow_i32(0, __x, __y, &__result);
        return __clamp_overflow<int32_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_i64(__x, __y);
#    elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int64_t __result;
        bool __overflow = _sub_overflow_i64(0, __x, __y, &__result);
        return __clamp_overflow<int64_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
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
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_u8(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint8_t __result;
        bool __overflow = _subborrow_u8(0, __x, __y, &__result);
        return __clamp_overflow<uint8_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_u16(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint16_t __result;
        bool __overflow = _subborrow_u16(0, __x, __y, &__result);
        return __clamp_overflow<uint16_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_u32(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint32_t __result;
        bool __overflow = _subborrow_u32(0, __x, __y, &__result);
        return __clamp_overflow<uint32_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
        return _sat_sub_u64(__x, __y);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        uint64_t __result;
        bool __overflow = _subborrow_u64(0, __x, __y, &__result);
        return __clamp_overflow<uint64_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
  }
#  endif // !_CCCL_COMPILER(NVRTC)
#  if _CCCL_HAS_CUDA_COMPILER
  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __impl_device(_Tp __x, _Tp __y) noexcept
  {
    if constexpr (_CCCL_TRAIT(is_signed, _Tp))
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        int16_t __result;
        asm("sub.s16 %0, %1, %2;" : "=h"(__result) : "h"(static_cast<int16_t>(__x)), "h"(static_cast<int16_t>(__y)));
        return static_cast<int8_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::min()),
          static_cast<int16_t>(_CUDA_VSTD::numeric_limits<int8_t>::max())));
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        int32_t __result = static_cast<int32_t>(__x) - static_cast<int32_t>(__y);
        return static_cast<int16_t>(_CUDA_VSTD::clamp(
          __result,
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::min()),
          static_cast<int32_t>(_CUDA_VSTD::numeric_limits<int16_t>::max())));
      }
      // Disabled due to nvbug 5033045
      // else if constexpr (sizeof(_Tp) == 4)
      // {
      //   int32_t __result{};
      //   asm("sub.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
      //   return __result;
      // }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
    else
    {
      return static_cast<_Tp>(_CUDA_VSTD::max<_Tp>(__x, __y) - __y);
    }
  }
#  endif // _CCCL_HAS_CUDA_COMPILER
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp sub_sat(_Tp __x, _Tp __y) noexcept
{
  using _Up = remove_cv_t<_Tp>;
#  if defined(_CCCL_BUILTIN_SUB_OVERFLOW)
  return __sub_sat::__impl_builtin<_Up>(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_SUB_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_SUB_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __sub_sat::__impl_host<_Up>(__x, __y);), (return __sub_sat::__impl_device<_Up>(__x, __y);))
  }
  return __sub_sat::__impl_generic<_Up>(__x, __y);
#  endif // !_CCCL_BUILTIN_SUB_OVERFLOW
}

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

#  if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __impl_builtin(_Tp __x, _Tp __y) noexcept
  {
    _Tp __result{};
    bool __overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__x, __y, &__result);

    return __clamp_overflow<_Tp>(__x, __y, __result, __overflow);
  }
#  endif // _CCCL_BUILTIN_MUL_OVERFLOW
#  if !_CCCL_COMPILER(NVRTC)
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

      if constexpr (sizeof(_Tp) == 1)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int16_t __result;
        bool __overflow = _mul_full_overflow_i8(__x, __y, &__result);
        return __clamp_overflow<int8_t>(__x, __y, static_cast<int8_t>(__result), __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int16_t __result;
        bool __overflow = _mul_overflow_i16(__x, __y, &__result);
        return __clamp_overflow<int16_t>(__x, __y, __result, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int32_t __result;
        bool __overflow = _mul_overflow_i32(__x, __y, &__result);
        return __clamp_overflow<int32_t>(__x, __y, __result, __overflow);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        const int64_t __result = __emul(__x, __y);
        return __clamp_overflow<int32_t>(
          __x,
          __y,
          static_cast<int32_t>(__result),
          __result > _CUDA_VSTD::numeric_limits<int32_t>::max()
            || __result < _CUDA_VSTD::numeric_limits<int32_t>::min());
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        int64_t __result;
        bool __overflow = _mul_overflow_i64(__x, __y, &__result);
        return __clamp_overflow<int64_t>(__x, __y, __result, __overflow);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        int64_t __hi = __mulh(__x, __y);
        int64_t __lo = __x * __y;
        return __clamp_overflow<int64_t>(__x, __y, __lo, __hi != _Tp{0} && __hi != _Tp{-1});
#    else
        return __impl_generic(__x, __y);
#    endif
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
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        uint16_t __result;
        bool __overflow = _mul_full_overflow_u8(__x, __y, &__result);
        return __clamp_overflow<uint8_t>(__x, __y, static_cast<uint8_t>(__result), __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        uint16_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u16(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint16_t>(__x, __y, __lo, __overflow);
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        uint32_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u32(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint32_t>(__x, __y, __lo, __overflow);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        const uint64_t __result = __emulu(__x, __y);
        return __clamp_overflow<uint32_t>(
          __x, __y, static_cast<uint32_t>(__result), __result > _CUDA_VSTD::numeric_limits<uint32_t>::max());
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
#    if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
        uint64_t __lo, __hi;
        bool __overflow = _mul_full_overflow_u64(__x, __y, &__lo, &__hi);
        return __clamp_overflow<uint64_t>(__x, __y, __lo, __overflow);
#    elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
        return __clamp_overflow<uint64_t>(__x, __y, __x * __y, __umulh(__x, __y) != _Tp{0});
#    else
        return __impl_generic(__x, __y);
#    endif
      }
      else
      {
        return __impl_generic(__x, __y);
      }
    }
  }
#  endif // !_CCCL_COMPILER(NVRTC)
#  if _CCCL_HAS_CUDA_COMPILER
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
#  endif // _CCCL_HAS_CUDA_COMPILER
};

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp mul_sat(_Tp __x, _Tp __y) noexcept
{
  using _Up = remove_cv_t<_Tp>;
#  if defined(_CCCL_BUILTIN_MUL_OVERFLOW)
  return __mul_sat::__impl_builtin<_Up>(__x, __y);
#  else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST, (return __mul_sat::__impl_host<_Up>(__x, __y);), (return __mul_sat::__impl_device<_Up>(__x, __y);))
  }
  return __mul_sat::__impl_generic<_Up>(__x, __y);
#  endif // !_CCCL_BUILTIN_MUL_OVERFLOW
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp div_sat(_Tp __x, _Tp __y) noexcept
{
  _CCCL_ASSERT(__y != _Tp{}, "division by zero");
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if (__x == _CUDA_VSTD::numeric_limits<_Tp>::min() && __y == _Tp{-1})
    {
      return _CUDA_VSTD::numeric_limits<_Tp>::max();
    }
  }
  return static_cast<_Tp>(__x / __y);
}

_CCCL_TEMPLATE(class _Up, class _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(__is_arithmetic_integral, _Up) _CCCL_AND _CCCL_TRAIT(__is_arithmetic_integral, _Tp))
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Up saturate_cast(_Tp __x) noexcept
{
  if (_CUDA_VSTD::cmp_less(__x, _CUDA_VSTD::numeric_limits<_Up>::min()))
  {
    return _CUDA_VSTD::numeric_limits<_Up>::min();
  }
  if (_CUDA_VSTD::cmp_greater(__x, _CUDA_VSTD::numeric_limits<_Up>::max()))
  {
    return _CUDA_VSTD::numeric_limits<_Up>::max();
  }
  return static_cast<_Up>(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2017

#endif // _LIBCUDACXX___NUMERIC_SATURATION_ARITHMETIC_H
