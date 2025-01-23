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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_COMPILER(MSVC)

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __msvc_constexpr_popc(_Tp __x) noexcept
{
  int __count = 0;
  for (int __i = 0; __i < numeric_limits<_Tp>::digits; ++__i)
  {
    __count += (__x & (_Tp{1} << __i)) ? 1 : 0;
  }
  return __count;
}

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI int __msvc_runtime_popc(_Tp __x) noexcept
{
#  if !defined(_M_ARM64)
  auto __ret = sizeof(_Tp) == sizeof(uint32_t) //
               ? __popcnt(static_cast<uint32_t>(__x))
               : __popcnt64(static_cast<uint64_t>(__x));
#  else
  auto __ret = sizeof(_Tp) == sizeof(uint32_t) //
               ? _CountOneBits(static_cast<uint32_t>(__x))
               : _CountOneBits64(static_cast<uint64_t>(__x));
#  endif
  return static_cast<int>(__ret);
}

#endif // !_CCCL_COMPILER(MSVC) ^^^

template <class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_popc(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
#if _CCCL_COMPILER(MSVC) && !defined(__CUDA_ARCH__) && !_CCCL_COMPILER(NVRTC)
  return is_constant_evaluated() ? _CUDA_VSTD::__msvc_constexpr_popc(__x) : _CUDA_VSTD::__msvc_runtime_popc(__x);
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return sizeof(_Tp) == sizeof(uint32_t)
         ? _CCCL_BUILTIN_POPCOUNT(static_cast<uint32_t>(__x))
         : _CCCL_BUILTIN_POPCOUNTLL(static_cast<uint64_t>(__x));
#endif // !_CCCL_COMPILER(MSVC) ^^^
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_POPC_H
