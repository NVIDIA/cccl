//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX__BIT_POPC_H
#define _LIBCUDACXX__BIT_POPC_H

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

#  if defined(_M_ARM64)
#    define _LIBCUDACXX_MSVC_POPC(x)   _CountOneBits(x)
#    define _LIBCUDACXX_MSVC_POPC64(x) _CountOneBits64(x)
#  else // ^^^ _M_ARM64 ^^^ / vvv !_M_ARM64 vvv
#    define _LIBCUDACXX_MSVC_POPC(x)   __popcnt(x)
#    define _LIBCUDACXX_MSVC_POPC64(x) __popcnt64(x)
#  endif // !_M_ARM64

#endif // _CCCL_COMPILER_MSVC

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __fallback_popc8(uint64_t __x)
{
  return static_cast<int>((__x * 0x0101010101010101) >> 56);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __fallback_popc16(uint64_t __x)
{
  return __fallback_popc8((__x + (__x >> 4)) & 0x0f0f0f0f0f0f0f0f);
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __fallback_popc32(uint64_t __x)
{
  return __fallback_popc16((__x & 0x3333333333333333) + ((__x >> 2) & 0x3333333333333333));
}
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __fallback_popc64(uint64_t __x)
{
  return __fallback_popc32(__x - ((__x >> 1) & 0x5555555555555555));
}

#if !defined(_CCCL_COMPILER_MSVC)

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popcount(uint32_t __x) noexcept
{
#  if defined(__CUDA_ARCH__)
  return __fallback_popc64(static_cast<uint64_t>(__x)); // no device constexpr builtins
#  else
  return __builtin_popcount(__x);
#  endif
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popcount(uint64_t __x) noexcept
{
#  if defined(__CUDA_ARCH__)
  return __fallback_popc64(static_cast<uint64_t>(__x)); // no device constexpr builtins
#  else
  return __builtin_popcountll(__x);
#  endif
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_popc(uint32_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __popc(__x);), (return __builtin_popcount(__x);))
  }
#  endif
  return __constexpr_popcount(static_cast<uint64_t>(__x));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_popc(uint64_t __x) noexcept
{
#  if _CCCL_STD_VER >= 2014
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __popcll(__x);), (return __builtin_popcountll(__x);))
  }
#  endif
  return __constexpr_popcount(static_cast<uint64_t>(__x));
}

#else // defined(_CCCL_COMPILER_MSVC)

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_popc(uint32_t __x)
{
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return static_cast<int>(_LIBCUDACXX_MSVC_POPC(__x));))
  }

  return __fallback_popc64(static_cast<uint64_t>(__x));
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __libcpp_popc(uint64_t __x)
{
  if (!__libcpp_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, (return static_cast<int>(_LIBCUDACXX_MSVC_POPC64(__x));))
  }

  return __fallback_popc64(static_cast<uint64_t>(__x));
}

#endif // MSVC

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_POPC_H
