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

#ifndef _LIBCUDACXX___NUMERIC_ADD_SAT_H
#define _LIBCUDACXX___NUMERIC_ADD_SAT_H

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
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp
__add_sat_clamp_overflow(_Tp, [[maybe_unused]] _Tp __y, _Tp __result, bool __overflow) noexcept
{
  if (__overflow)
  {
    if constexpr (_CCCL_TRAIT(is_unsigned, _Tp))
    {
      __result = numeric_limits<_Tp>::max();
    }
    else
    {
      __result = (__y < _Tp{0}) ? numeric_limits<_Tp>::min() : numeric_limits<_Tp>::max();
    }
  }
  return __result;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __add_sat_impl_generic(_Tp __x, _Tp __y) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    using _Up     = make_unsigned_t<_Tp>;
    auto __result = static_cast<_Tp>(static_cast<_Up>(__x) + static_cast<_Up>(__y));
    return _CUDA_VSTD::__add_sat_clamp_overflow(__x, __y, __result, (__result < __x) == !(__y < static_cast<_Tp>(0)));
  }
  else
  {
    auto __result = static_cast<_Tp>(__x + __y);
    return _CUDA_VSTD::__add_sat_clamp_overflow(__x, __y, __result, (__result < __x));
  }
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST _Tp __add_sat_impl_host(_Tp __x, _Tp __y) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(int8_t))
    {
      return ::_sat_add_i8(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int16_t))
    {
      return ::_sat_add_i16(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      return ::_sat_add_i32(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      return ::_sat_add_i64(__x, __y);
    }
    else
    {
      return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
    }
#  elif _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(int8_t))
    {
      int8_t __result;
      bool __overflow = ::_add_overflow_i8(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<int8_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int16_t))
    {
      int16_t __result;
      bool __overflow = ::_add_overflow_i16(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<int16_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      int32_t __result;
      bool __overflow = ::_add_overflow_i32(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<int32_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      int64_t __result;
      bool __overflow = ::_add_overflow_i64(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<int64_t>(__x, __y, __result, __overflow);
    }
    else
    {
      return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64) ^^^ / vvv _CCCL_COMPILER(MSVC, <, 19, 37) ||
        // !_CCCL_ARCH(X86_64) vvv
    return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
#  endif // ^^^ _CCCL_COMPILER(MSVC, <, 19, 37) || !_CCCL_ARCH(X86_64) ^^^
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 41) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      return ::_sat_add_u8(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      return ::_sat_add_u16(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      return ::_sat_add_u32(__x, __y);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      return ::_sat_add_u64(__x, __y);
    }
    else
    {
      return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
    }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      uint8_t __result;
      bool __overflow = ::_addcarry_u8(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<uint8_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      uint16_t __result;
      bool __overflow = ::_addcarry_u16(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<uint16_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      uint32_t __result;
      bool __overflow = ::_addcarry_u32(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<uint32_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      uint64_t __result;
      bool __overflow = ::_addcarry_u64(0, __x, __y, &__result);
      return _CUDA_VSTD::__add_sat_clamp_overflow<uint64_t>(__x, __y, __result, __overflow);
    }
    else
    {
      return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
  } // ^^^ unsigned types ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_HAS_CUDA_COMPILER()
template <class _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __add_sat_impl_device(_Tp __x, _Tp __y) noexcept
{
  if constexpr (_CCCL_TRAIT(is_signed, _Tp))
  {
    if constexpr (sizeof(_Tp) < sizeof(int32_t))
    {
      constexpr auto __min = int32_t{numeric_limits<_Tp>::min()};
      constexpr auto __max = int32_t{numeric_limits<_Tp>::max()};
      auto __result        = static_cast<int32_t>(__x) + static_cast<int32_t>(__y);
      return static_cast<_Tp>(::min(::max(__result, __min), __max));
    }
    else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      int32_t __result{};
      asm("add.sat.s32 %0, %1, %2;" : "=r"(__result) : "r"(__x), "r"(__y));
      return __result;
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      int64_t __result{};
      asm volatile(
        "{\n\t"
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
      return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
    }
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
    using _Up = _If<sizeof(_Tp) < sizeof(uint32_t), uint32_t, _Tp>;
    if constexpr (is_same_v<uint16_t, _Tp>)
    {
      const auto bneg_x = static_cast<_Tp>(~static_cast<_Up>(__x));
      printf("static_cast<_Up>(__x):                    %d\n", static_cast<_Up>(__x));
      printf("static_cast<_Tp>(~static_cast<_Up>(__x)): %d\n", static_cast<_Tp>(~static_cast<_Up>(__x)));
      printf("_CUDA_VSTD::min(__y, bneg_x):             %d\n", _CUDA_VSTD::min(__y, bneg_x));
      printf("res:                                      %d\n", static_cast<_Tp>(__x + _CUDA_VSTD::min(__y, bneg_x)));
    }
    const auto __bneg_x = static_cast<_Tp>(~static_cast<_Up>(__x));
    return static_cast<_Tp>(__x + _CUDA_VSTD::min(__y, __bneg_x));
  } // ^^^ unsigned types ^^^
}
#endif // _CCCL_HAS_CUDA_COMPILER()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp add_sat(_Tp __x, _Tp __y) noexcept
{
#if defined(_CCCL_BUILTIN_ADD_OVERFLOW) && !defined(__CUDA_ARCH__)
  _Tp __result{};
  bool __overflow = _CCCL_BUILTIN_ADD_OVERFLOW(__x, __y, &__result);
  return _CUDA_VSTD::__add_sat_clamp_overflow(__x, __y, __result, __overflow);
#else // ^^^ _CCCL_BUILTIN_ADD_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_ADD_OVERFLOW vvv
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return _CUDA_VSTD::__add_sat_impl_host(__x, __y);),
                      (return _CUDA_VSTD::__add_sat_impl_device(__x, __y);))
  }
  return _CUDA_VSTD::__add_sat_impl_generic(__x, __y);
#endif // !_CCCL_BUILTIN_ADD_OVERFLOW
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___NUMERIC_ADD_SAT_H
