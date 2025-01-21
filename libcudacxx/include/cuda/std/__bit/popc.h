//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popc(uint32_t __x) noexcept
{
  // no device constexpr builtins
  int __count = 0;
  for (int __i = 0; __i < 32; ++__i)
  {
    __count += (__x & (uint32_t{1} << __i)) ? 1 : 0;
  }
  return __count;
}

_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_popc(uint64_t __x) noexcept
{
  int __count = 0;
  for (int __i = 0; __i < 64; ++__i)
  {
    __count += (__x & (uint64_t{1} << __i)) ? 1 : 0;
  }
  return __count;
}

// constexpr is required for GCC8
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __runtime_popc(uint32_t __x) noexcept
{
#if defined(__CUDA_ARCH__)
  return ::__popc(__x);
#elif _CCCL_COMPILER(MSVC) && !defined(_M_ARM64) // _CCCL_COMPILER(MSVC) + X86 vvv
  return static_cast<int>(::__popcnt(__x));
#elif _CCCL_COMPILER(MSVC) && defined(_M_ARM64) // _CCCL_COMPILER(MSVC) + X86 vvv
  return static_cast<int>(::_CountOneBits(__x));
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return ::__builtin_popcount(__x);
#endif // !_CCCL_COMPILER(MSVC) ^^^
}

// constexpr is required for GCC8
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __runtime_popc(uint64_t __x) noexcept
{
#if defined(__CUDA_ARCH__)
  return ::__popcll(__x);
#elif _CCCL_COMPILER(MSVC) && !defined(_M_ARM64) // _CCCL_COMPILER(MSVC) + X86 vvv
  return static_cast<int>(::__popcnt64(__x));
#elif _CCCL_COMPILER(MSVC) && defined(_M_ARM64) // _CCCL_COMPILER(MSVC) + ARM64 vvv
  return static_cast<int>(::_CountOneBits64(__x));
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return ::__builtin_popcountll(__x);
#endif // !_CCCL_COMPILER(MSVC) ^^^
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_popc(uint32_t __x) noexcept
{
  return is_constant_evaluated() ? _CUDA_VSTD::__constexpr_popc(__x) : _CUDA_VSTD::__runtime_popc(__x);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_popc(uint64_t __x) noexcept
{
  return is_constant_evaluated() ? _CUDA_VSTD::__constexpr_popc(__x) : _CUDA_VSTD::__runtime_popc(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_POPC_H
