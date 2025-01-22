//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#if _CCCL_COMPILER(MSVC)
#  include <intrin.h>
#endif // _CCCL_COMPILER(MSVC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr int __constexpr_ctz(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
  for (int __i = 0; __i < numeric_limits<_Tp>::digits; ++__i)
  {
    if (__x & (_Tp{1} << __i))
    {
      return __i;
    }
  }
  return numeric_limits<_Tp>::digits;
}

#if _CCCL_COMPILER(MSVC)

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI int __msvc_runtime_ctz(_Tp __x) noexcept
{
  unsigned long __where;
  auto __res = sizeof(_Tp) == sizeof(uint32_t)
               ? _BitScanForward(&__where, static_cast<uint32_t>(__x))
               : _BitScanForward64(&__where, static_cast<uint64_t>(__x));
  return (__res) ? static_cast<int>(__where) : numeric_limits<_Tp>::digits;
}

#endif // _CCCL_COMPILER(MSVC)

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr int __cccl_ctz(_Tp __x) noexcept
{
  static_assert(is_same_v<_Tp, uint32_t> || is_same_v<_Tp, uint64_t>);
#if defined(__CUDA_ARCH__)
  if (is_constant_evaluated())
  {
    return _CUDA_VSTD::__constexpr_ctz(__x);
  }
  else
  {
    return sizeof(_Tp) == sizeof(uint32_t)
           ? __clz(__brev(static_cast<uint32_t>(__x)))
           : __clzll(__brevll(static_cast<uint64_t>(__x)));
  }
#elif _CCCL_COMPILER(MSVC)
  return is_constant_evaluated() ? _CUDA_VSTD::__constexpr_ctz(__x) : _CUDA_VSTD::__msvc_runtime_ctz(__x);
#else // _CCCL_COMPILER(MSVC) ^^^ / !_CCCL_COMPILER(MSVC) vvv
  return sizeof(_Tp) == sizeof(uint32_t)
         ? _CCCL_BUILTIN_CTZ(static_cast<uint32_t>(__x))
         : _CCCL_BUILTIN_CTZLL(static_cast<uint64_t>(__x));
#endif
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX__BIT_CTZ_H
