//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__BIT_CLZ_H
#define _LIBCUDACXX__BIT_CLZ_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstdint>

#if defined(_CCCL_COMPILER_MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER_MSVC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz2(uint64_t __x, int __c)
{
  return !!(~__x & 0x2) ^ __c;
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz4(uint64_t __x, int __c)
{
  return __binary_clz2(__x >> 2 * !!(__x & 0xC), __c + 2 * !(__x & 0xC));
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz8(uint64_t __x, int __c)
{
  return __binary_clz4(__x >> 4 * !!(__x & 0xF0), __c + 4 * !(__x & 0xF0));
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz16(uint64_t __x, int __c)
{
  return __binary_clz8(__x >> 8 * !!(__x & 0xFF00), __c + 8 * !(__x & 0xFF00));
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz32(uint64_t __x, int __c)
{
  return __binary_clz16(__x >> 16 * !!(__x & 0xFFFF0000), __c + 16 * !(__x & 0xFFFF0000));
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __binary_clz64(uint64_t __x)
{
  return __binary_clz32(__x >> 32 * !!(__x & 0xFFFFFFFF00000000), 32 * !(__x & 0xFFFFFFFF00000000));
}

#if !defined(_CCCL_COMPILER_MSVC)

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz(uint32_t __x) noexcept
{
#  if defined(__CUDA_ARCH__)
  return __binary_clz32(static_cast<uint64_t>(__x), 0); // no device constexpr builtins
#  else
  return __builtin_clz(__x);
#  endif
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz(uint64_t __x) noexcept
{
#  if defined(__CUDA_ARCH__)
  return __binary_clz64(__x); // no device constexpr builtins
#  else
  return __builtin_clzll(__x);
#  endif
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_clz(uint32_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __clz(__x);), (return __builtin_clz(__x);))
  }
#  endif
  return __constexpr_clz(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_clz(uint64_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __clzll(__x);), (return __builtin_clzll(__x);))
  }
#  endif
  return __constexpr_clz(__x);
}

#else // defined(_CCCL_COMPILER_MSVC)

// Precondition:  __x != 0
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_clz(uint32_t __x)
{
#  if !defined(__CUDA_ARCH__)
  if (!__libcpp_default_is_constant_evaluated())
  {
    unsigned long __where = 0;
    if (_BitScanReverse(&__where, __x))
    {
      return static_cast<int>(31 - __where);
    }
    return 32; // Undefined Behavior.
  }
#  endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED && !defined(__CUDA_ARCH__)

  return __binary_clz32(static_cast<uint64_t>(__x), 0);
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_clz(uint64_t __x)
{
#  if !defined(__CUDA_ARCH__)
  if (!__libcpp_default_is_constant_evaluated())
  {
    unsigned long __where = 0;
#    if defined(_LIBCUDACXX_HAS_BITSCAN64)
    if (_BitScanReverse64(&__where, __x))
    {
      return static_cast<int>(63 - __where);
    }
#    else
    // Win32 doesn't have _BitScanReverse64 so emulate it with two 32 bit calls.
    if (_BitScanReverse(&__where, static_cast<uint32_t>(__x >> 32)))
    {
      return static_cast<int>(63 - (__where + 32));
    }
    if (_BitScanReverse(&__where, static_cast<uint32_t>(__x)))
    {
      return static_cast<int>(63 - __where);
    }
#    endif
    return 64; // Undefined Behavior.
  }
#  endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED && !defined(__CUDA_ARCH__)

  return __binary_clz64(static_cast<uint64_t>(__x));
}

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_CLZ_H
