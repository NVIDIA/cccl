//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_BIT_FALLBACKS_H
#define __CCCL_BIT_FALLBACKS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz2(uint64_t __x, int __c) noexcept
{
  return __c + !(__x & 0x1);
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz4(uint64_t __x, int __c) noexcept
{
  return __binary_ctz2(__x >> 2 * !(__x & 0x3), __c + 2 * !(__x & 0x3));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz8(uint64_t __x, int __c) noexcept
{
  return __binary_ctz4(__x >> 4 * !(__x & 0x0F), __c + 4 * !(__x & 0x0F));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz16(uint64_t __x, int __c) noexcept
{
  return __binary_ctz8(__x >> 8 * !(__x & 0x00FF), __c + 8 * !(__x & 0x00FF));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz32(uint64_t __x, int __c) noexcept
{
  return __binary_ctz16(__x >> 16 * !(__x & 0x0000FFFF), __c + 16 * !(__x & 0x0000FFFF));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_ctz64(uint64_t __x) noexcept
{
  return __binary_ctz32(__x >> 32 * !(__x & 0x00000000FFFFFFFF), 32 * !(__x & 0x00000000FFFFFFFF));
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz2(uint64_t __x, int __c)
{
  return !!(~__x & 0x2) ^ __c;
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz4(uint64_t __x, int __c)
{
  return __binary_clz2(__x >> 2 * !!(__x & 0xC), __c + 2 * !(__x & 0xC));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz8(uint64_t __x, int __c)
{
  return __binary_clz4(__x >> 4 * !!(__x & 0xF0), __c + 4 * !(__x & 0xF0));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz16(uint64_t __x, int __c)
{
  return __binary_clz8(__x >> 8 * !!(__x & 0xFF00), __c + 8 * !(__x & 0xFF00));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz32(uint64_t __x, int __c)
{
  return __binary_clz16(__x >> 16 * !!(__x & 0xFFFF0000), __c + 16 * !(__x & 0xFFFF0000));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __binary_clz64(uint64_t __x)
{
  return __binary_clz32(__x >> 32 * !!(__x & 0xFFFFFFFF00000000), 32 * !(__x & 0xFFFFFFFF00000000));
}

inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __fallback_popc8(uint64_t __x)
{
  return static_cast<int>((__x * 0x0101010101010101) >> 56);
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __fallback_popc16(uint64_t __x)
{
  return __fallback_popc8((__x + (__x >> 4)) & 0x0f0f0f0f0f0f0f0f);
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __fallback_popc32(uint64_t __x)
{
  return __fallback_popc16((__x & 0x3333333333333333) + ((__x >> 2) & 0x3333333333333333));
}
inline _LIBCUDACXX_INLINE_VISIBILITY constexpr int __fallback_popc64(uint64_t __x)
{
  return __fallback_popc32(__x - ((__x >> 1) & 0x5555555555555555));
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif //__CCCL_BIT_FALLBACKS_H
