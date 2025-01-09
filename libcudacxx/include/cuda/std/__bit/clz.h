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

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz(uint32_t __x) noexcept
{
  for (int __i = 31; __i >= 0; --__i)
  {
    if (__x & (uint32_t{1} << __i))
    {
      return 31 - __i;
    }
  }
  return 32;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_clz(uint64_t __x) noexcept
{
  for (int __i = 63; __i >= 0; --__i)
  {
    if (__x & (uint64_t{1} << __i))
    {
      return 63 - __i;
    }
  }
  return 64;
}

_LIBCUDACXX_HIDE_FROM_ABI int __runtime_clz(uint32_t __x) noexcept
{
#if defined(__CUDA_ARCH__)
  return ::__clz(__x);
#elif _CCCL_COMPILER(MSVC) // _CCCL_COMPILER(MSVC) vvv
  unsigned long __where = 0;
  if (::_BitScanReverse32(&__where, __x))
  {
    return 31 - static_cast<int>(__where);
  }
  return 32; // Undefined Behavior.
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return ::__builtin_clz(__x);
#endif // _CCCL_COMPILER(MSVC)
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __runtime_clz(uint64_t __x) noexcept
{
#if defined(__CUDA_ARCH__)
  return ::__clzll(__x);
#elif _CCCL_COMPILER(MSVC) // _CCCL_COMPILER(MSVC) vvv
  unsigned long __where = 0;
  if (::_BitScanReverse64(&__where, __x))
  {
    return 63 - static_cast<int>(__where);
  }
  return 64; // Undefined Behavior.
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return ::__builtin_clzll(__x);
#endif // !_CCCL_COMPILER(MSVC) ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_clz(uint32_t __x) noexcept
{
  return _CUDA_VSTD::is_constant_evaluated() ? _CUDA_VSTD::__constexpr_clz(__x) : _CUDA_VSTD::__runtime_clz(__x);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_clz(uint64_t __x) noexcept
{
  return _CUDA_VSTD::is_constant_evaluated() ? _CUDA_VSTD::__constexpr_clz(__x) : _CUDA_VSTD::__runtime_clz(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_CLZ_H
