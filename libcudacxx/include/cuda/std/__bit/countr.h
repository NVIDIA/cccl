//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_COUNTR_H
#define _CUDA_STD___BIT_COUNTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(builtin_ctz) || _CCCL_COMPILER(GCC, <, 10) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)
#  define _CCCL_BUILTIN_CTZ(...)   __builtin_ctz(__VA_ARGS__)
#  define _CCCL_BUILTIN_CTZLL(...) __builtin_ctzll(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ctz)

#if _CCCL_CHECK_BUILTIN(builtin_ctzg)
#  define _CCCL_BUILTIN_CTZG(...) __builtin_ctzg(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(builtin_ctzg)

// nvcc doesn't support this builtin in device code and before 13.0 not even in host code
#if (_CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()) || _CCCL_CUDA_COMPILER(NVCC, <, 13)
#  undef _CCCL_BUILTIN_CTZG
#endif // (_CCCL_CUDA_COMPILER(NVCC) && _CCCL_DEVICE_COMPILATION()) || _CCCL_CUDA_COMPILER(NVCC, <, 13)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if !defined(_CCCL_BUILTIN_CTZG)

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_countr_zero_impl_constexpr(_Tp __v) noexcept
{
  constexpr auto __digits = numeric_limits<uint32_t>::digits;

  if (__v == 0)
  {
    return __digits;
  }

  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    if (__v & 1)
    {
      return 0;
    }
    int __pos = 1;
    for (int __i = __digits / 2; __i >= 2; __i /= 2)
    {
      const auto __mark = ~uint32_t{0} >> (__digits - __i);
      if ((__v & __mark) == 0)
      {
        __v >>= __i;
        __pos += __i;
      }
    }
    return (__pos - (__v & 1));
  }
  else
  {
    const auto __hi = static_cast<uint32_t>(__v >> 32);
    const auto __lo = static_cast<uint32_t>(__v);
    return (__lo != 0) ? ::cuda::std::__cccl_countr_zero_impl_constexpr(__lo)
                       : numeric_limits<uint32_t>::digits + ::cuda::std::__cccl_countr_zero_impl_constexpr(__hi);
  }
}

#  if !_CCCL_COMPILER(NVRTC)
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_HOST int __cccl_countr_zero_impl_host(_Tp __v) noexcept
{
  // nvcc does not support __builtin_ctz, so we use it only for host code
#    if defined(_CCCL_BUILTIN_CTZ)
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return _CCCL_BUILTIN_CTZ(__v);
  }
  else
  {
    return _CCCL_BUILTIN_CTZLL(__v);
  }
#    elif _CCCL_COMPILER(MSVC)
  unsigned long __where{};
  unsigned char __res{};
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    __res = ::_BitScanForward(&__where, __v);
  }
  else
  {
    __res = ::_BitScanForward64(&__where, __v);
  }
  return (__res) ? static_cast<int>(__where) : numeric_limits<_Tp>::digits;
#    else
  return ::cuda::std::__cccl_countr_zero_impl_constexpr(__v);
#    endif // _CCCL_COMPILER(MSVC)
}
#  endif // !_CCCL_COMPILER(NVRTC)

#  if _CCCL_CUDA_COMPILATION()
template <typename _Tp>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE int __cccl_countr_zero_impl_device(_Tp __v) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(uint32_t))
  {
    return static_cast<int>(::__clz(static_cast<int>(::__brev(__v))));
  }
  else
  {
    return static_cast<int>(::__clzll(static_cast<long long>(::__brevll(__v))));
  }
}
#  endif // _CCCL_CUDA_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr int __cccl_countr_zero_impl(_Tp __v) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
  _CCCL_IF_NOT_CONSTEVAL_DEFAULT
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return ::cuda::std::__cccl_countr_zero_impl_host(__v);),
                      (return ::cuda::std::__cccl_countr_zero_impl_device(__v);));
  }
  return ::cuda::std::__cccl_countr_zero_impl_constexpr(__v);
}

#endif // !_CCCL_BUILTIN_CTZG

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int countr_zero(_Tp __v) noexcept
{
  int __count{};

#if defined(_CCCL_BUILTIN_CTZG)
  __count = _CCCL_BUILTIN_CTZG(__v, numeric_limits<_Tp>::digits);
#else // ^^^ __CCCL_BUILTIN_CTZG ^^^ / vvv !__CCCL_BUILTIN_CTZG vvv
  if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
  {
    using _Sp = _If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
    __count   = (__v != 0) ? ::cuda::std::__cccl_countr_zero_impl(static_cast<_Sp>(__v)) : numeric_limits<_Tp>::digits;
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (size_t __i = 0; __i < sizeof(_Tp) / sizeof(uint64_t); ++__i)
    {
      const auto __value64 = static_cast<uint64_t>(__v);
      if (__value64 != 0)
      {
        __count += ::cuda::std::countr_zero(__value64);
        break;
      }
      __count += numeric_limits<uint64_t>::digits;
      __v >>= numeric_limits<uint64_t>::digits;
    }
  }
#endif // ^^^ !__CCCL_BUILTIN_CTZG ^^^

  _CCCL_ASSUME(__count >= 0 && __count <= numeric_limits<_Tp>::digits);
  return __count;
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int countr_one(_Tp __t) noexcept
{
  return ::cuda::std::countr_zero(static_cast<_Tp>(~__t));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_COUNTR_H
