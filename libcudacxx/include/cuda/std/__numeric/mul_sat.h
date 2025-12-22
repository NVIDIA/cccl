//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___NUMERIC_MUL_SAT_H
#define _CUDA_STD___NUMERIC_MUL_SAT_H

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
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <nv/target>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// todo: refactor this and use cuda::mul_overflow

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp
__mul_sat_clamp_overflow([[maybe_unused]] _Tp __x, [[maybe_unused]] _Tp __y, _Tp __result, bool __overflow) noexcept
{
  if (__overflow)
  {
    if constexpr (is_unsigned_v<_Tp>)
    {
      __result = numeric_limits<_Tp>::max();
    }
    else
    {
      __result = (static_cast<_Tp>(__x ^ __y) >= _Tp{0}) ? numeric_limits<_Tp>::max() : numeric_limits<_Tp>::min();
    }
  }
  return __result;
}

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr _Tp __mul_sat_impl_generic(_Tp __x, _Tp __y) noexcept
{
  if (__x == _Tp{0} || __y == _Tp{0})
  {
    return _Tp{0};
  }

  if constexpr (is_signed_v<_Tp>)
  {
    if (__x == _Tp{-1})
    {
      if (__y == numeric_limits<_Tp>::min())
      {
        return numeric_limits<_Tp>::max();
      }
      return static_cast<_Tp>(-__y);
    }

    if (__y == _Tp{-1})
    {
      if (__x == numeric_limits<_Tp>::min())
      {
        return numeric_limits<_Tp>::max();
      }
      return static_cast<_Tp>(-__x);
    }

    if (__x > _Tp{0} && __y > _Tp{0})
    {
      if (__x > static_cast<_Tp>(numeric_limits<_Tp>::max() / __y))
      {
        return numeric_limits<_Tp>::max();
      }
    }
    else if (__x < _Tp{0} && __y < _Tp{0})
    {
      if (__x < static_cast<_Tp>(numeric_limits<_Tp>::max() / __y))
      {
        return numeric_limits<_Tp>::max();
      }
    }
    else if (__x < _Tp{0} && __y > _Tp{0})
    {
      if (__x < static_cast<_Tp>(numeric_limits<_Tp>::min() / __y))
      {
        return numeric_limits<_Tp>::min();
      }
    }
    else
    {
      if (__x > static_cast<_Tp>(numeric_limits<_Tp>::min() / __y))
      {
        return numeric_limits<_Tp>::min();
      }
    }
  }
  else
  {
    if (__x > static_cast<_Tp>(numeric_limits<_Tp>::max() / __y))
    {
      return numeric_limits<_Tp>::max();
    }
  }
  return static_cast<_Tp>(__x * __y);
}

#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp __mul_sat_impl_host(_Tp __x, _Tp __y) noexcept
{
  if constexpr (is_signed_v<_Tp>)
  {
    if ((__x == _Tp{-1} && __y == numeric_limits<_Tp>::min()) || (__y == _Tp{-1} && __x == numeric_limits<_Tp>::min()))
    {
      return numeric_limits<_Tp>::max();
    }

#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(int8_t))
    {
      int16_t __result;
      bool __overflow = ::_mul_full_overflow_i8(__x, __y, &__result);
      return ::cuda::std::__mul_sat_clamp_overflow<int8_t>(__x, __y, static_cast<int8_t>(__result), __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int16_t))
    {
      int16_t __result;
      bool __overflow = ::_mul_overflow_i16(__x, __y, &__result);
      return ::cuda::std::__mul_sat_clamp_overflow<int16_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      int32_t __result;
      bool __overflow = ::_mul_overflow_i32(__x, __y, &__result);
      return ::cuda::std::__mul_sat_clamp_overflow<int32_t>(__x, __y, __result, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      int64_t __result;
      bool __overflow = ::_mul_overflow_i64(__x, __y, &__result);
      return ::cuda::std::__mul_sat_clamp_overflow<int64_t>(__x, __y, __result, __overflow);
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(int32_t))
    {
      const int64_t __result = ::__emul(__x, __y);
      return ::cuda::std::__mul_sat_clamp_overflow<int32_t>(
        __x,
        __y,
        static_cast<int32_t>(__result),
        __result > numeric_limits<int32_t>::max() || __result < numeric_limits<int32_t>::min());
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      int64_t __hi = ::__mulh(__x, __y);
      int64_t __lo = __x * __y;
      return ::cuda::std::__mul_sat_clamp_overflow<int64_t>(__x, __y, __lo, __hi != _Tp{0} && __hi != _Tp{-1});
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    return ::cuda::std::__mul_sat_impl_generic(__x, __y);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
#  if _CCCL_COMPILER(MSVC, >=, 19, 37) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(uint8_t))
    {
      uint16_t __result;
      bool __overflow = ::_mul_full_overflow_u8(__x, __y, &__result);
      return ::cuda::std::__mul_sat_clamp_overflow<uint8_t>(__x, __y, static_cast<uint8_t>(__result), __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint16_t))
    {
      uint16_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u16(__x, __y, &__lo, &__hi);
      return ::cuda::std::__mul_sat_clamp_overflow<uint16_t>(__x, __y, __lo, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      uint32_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u32(__x, __y, &__lo, &__hi);
      return ::cuda::std::__mul_sat_clamp_overflow<uint32_t>(__x, __y, __lo, __overflow);
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      uint64_t __lo, __hi;
      bool __overflow = ::_mul_full_overflow_u64(__x, __y, &__lo, &__hi);
      return ::cuda::std::__mul_sat_clamp_overflow<uint64_t>(__x, __y, __lo, __overflow);
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
#  elif _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64)
    if constexpr (sizeof(_Tp) == sizeof(uint32_t))
    {
      const uint64_t __result = ::__emulu(__x, __y);
      return ::cuda::std::__mul_sat_clamp_overflow<uint32_t>(
        __x, __y, static_cast<uint32_t>(__result), __result > numeric_limits<uint32_t>::max());
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      return ::cuda::std::__mul_sat_clamp_overflow<uint64_t>(__x, __y, __x * __y, ::__umulh(__x, __y) != _Tp{0});
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
#  else // ^^^ _CCCL_COMPILER(MSVC) && _CCCL_ARCH(X86_64) ^^^ / vvv !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) vvv
    return ::cuda::std::__mul_sat_impl_generic(__x, __y);
#  endif // ^^^ !_CCCL_COMPILER(MSVC) || !_CCCL_ARCH(X86_64) ^^^
  } // ^^^ unsigned types ^^^
}
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp __mul_sat_impl_device(_Tp __x, _Tp __y) noexcept
{
  if constexpr (is_signed_v<_Tp>)
  {
    if constexpr (sizeof(_Tp) <= sizeof(int32_t))
    {
      using _Up    = __make_nbit_int_t<2 * sizeof(_Tp) * CHAR_BIT>;
      _Up __result = static_cast<_Up>(__x) * static_cast<_Up>(__y);
      return static_cast<_Tp>(::cuda::std::clamp(
        __result, static_cast<_Up>(numeric_limits<_Tp>::min()), static_cast<_Up>(numeric_limits<_Tp>::max())));
    }
    else if constexpr (sizeof(_Tp) == sizeof(int64_t))
    {
      if ((__x == _Tp{-1} && __y == numeric_limits<_Tp>::min())
          || (__y == _Tp{-1} && __x == numeric_limits<_Tp>::min()))
      {
        return numeric_limits<_Tp>::max();
      }

      int64_t __lo, __hi;
      __lo = __x * __y;
      __hi = ::__mul64hi(__x, __y);
      return ::cuda::std::__mul_sat_clamp_overflow<int64_t>(__x, __y, __lo, __hi != _Tp{0} && __hi != _Tp{-1});
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
  }
  else // ^^^ signed types ^^^ / vvv unsigned types vvv
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
    {
      using _Up    = __make_nbit_uint_t<2 * sizeof(_Tp) * CHAR_BIT>;
      _Up __result = static_cast<_Up>(__x) * static_cast<_Up>(__y);
      return static_cast<_Tp>(::cuda::std::min(__result, static_cast<_Up>(numeric_limits<_Tp>::max())));
    }
    else if constexpr (sizeof(_Tp) == sizeof(uint64_t))
    {
      uint64_t __lo, __hi;
      __lo = __x * __y;
      __hi = ::__umul64hi(__x, __y);
      return (__hi == uint64_t{0}) ? __lo : numeric_limits<_Tp>::max();
    }
    else
    {
      return ::cuda::std::__mul_sat_impl_generic(__x, __y);
    }
  } // ^^^ unsigned types ^^^
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp mul_sat(_Tp __x, _Tp __y) noexcept
{
#if defined(_CCCL_BUILTIN_MUL_OVERFLOW) && 0
  _Tp __result{};
  bool __overflow = _CCCL_BUILTIN_MUL_OVERFLOW(__x, __y, &__result);
  return ::cuda::std::__mul_sat_clamp_overflow(__x, __y, __result, __overflow);
#else // ^^^ _CCCL_BUILTIN_MUL_OVERFLOW ^^^ / vvv !_CCCL_BUILTIN_MUL_OVERFLOW vvv
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return ::cuda::std::__mul_sat_impl_host(__x, __y);),
                      (return ::cuda::std::__mul_sat_impl_device(__x, __y);))
  }
  return ::cuda::std::__mul_sat_impl_generic(__x, __y);
#endif // !_CCCL_BUILTIN_MUL_OVERFLOW
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___NUMERIC_MUL_SAT_H
