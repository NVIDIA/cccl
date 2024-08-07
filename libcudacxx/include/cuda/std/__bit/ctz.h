//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__BIT_CTZ_H
#define _LIBCUDACXX__BIT_CTZ_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/fallbacks.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstdint>

#if defined(_CCCL_COMPILER_MSVC)
#  include <intrin.h>
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if !defined(_CCCL_COMPILER_MSVC)

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __constexpr_ctz(uint32_t __x) noexcept
{
  NV_IF_TARGET(NV_IS_DEVICE,
               (return __binary_ctz32(static_cast<uint64_t>(__x), 0);), // no device constexpr builtins
               (return __builtin_ctz(__x);))
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __constexpr_ctz(uint64_t __x) noexcept
{
  NV_IF_TARGET(NV_IS_DEVICE,
               (return __binary_ctz64(__x);), // no device constexpr builtins
               (return __builtin_ctzll(__x);))
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __libcpp_ctz(uint32_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE, (return (!__x) ? (sizeof(uint32_t) * 8) : (__ffs(__x) - 1);), (return __builtin_ctz(__x);))
  }
#  endif
  return __constexpr_ctz(__x);
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __libcpp_ctz(uint64_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_IS_DEVICE, (return (!__x) ? (sizeof(uint64_t) * 8) : (__ffsll(__x) - 1);), (return __builtin_ctzll(__x);))
  }
#  endif
  return __constexpr_ctz(__x);
}

#else // defined(_CCCL_COMPILER_MSVC)

// Precondition:  __x != 0
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __libcpp_ctz(uint32_t __x)
{
#  if !defined(__CUDA_ARCH__)
  if (!__libcpp_is_constant_evaluated())
  {
    unsigned long __where = 0;
    if (_BitScanForward(&__where, __x))
    {
      return static_cast<int>(__where);
    }
    return 32;
  }
#  endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED && !defined(__CUDA_ARCH__)

  return __binary_ctz32(static_cast<uint64_t>(__x), 0);
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __libcpp_ctz(uint64_t __x)
{
#  if !defined(__CUDA_ARCH__)
  if (!__libcpp_is_constant_evaluated())
  {
    unsigned long __where = 0;
#    if defined(_LIBCUDACXX_HAS_BITSCAN64) && (defined(_M_AMD64) || defined(__x86_64__))
    if (_BitScanForward64(&__where, __x))
    {
      return static_cast<int>(__where);
    }
#    else
    // Win32 doesn't have _BitScanForward64 so emulate it with two 32 bit calls.
    if (_BitScanForward(&__where, static_cast<uint32_t>(__x)))
    {
      return static_cast<int>(__where);
    }
    if (_BitScanForward(&__where, static_cast<uint32_t>(__x >> 32)))
    {
      return static_cast<int>(__where + 32);
    }
#    endif
    return 64;
  }
#  endif // _LIBCUDACXX_IS_CONSTANT_EVALUATED && !defined(__CUDA_ARCH__)

  return __binary_ctz64(__x);
}

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_CTZ_H
