//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_FNS_H
#define _CUDA___BIT_BIT_FNS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Finds the position of the set bit with rank \p __rank in \p __value, counting set bits
//! from the least significant one.
//!
//! @param[in] __value The unsigned integer value to search.
//! @param[in] __rank The zero-based rank of the set bit to find.
//! @return The zero-based position of the selected set bit, or `-1` (`0xFFFFFFFF`, the not-found
//!         result of @c __fns) if \p __value has fewer than `__rank + 1` set bits.
//! @pre `0 <= __rank && __rank < num_bits(_Tp)`.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int bit_fns(const _Tp __value, const int __rank) noexcept
{
  constexpr int __digits = ::cuda::std::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__rank >= 0 && __rank < __digits, "cuda::bit_fns: rank out of range");
  if (__rank >= ::cuda::std::popcount(__value))
  {
    return -1;
  }
  auto __window   = +__value; // small types are promoted to 32 bits
  int __remaining = __rank;
  int __position  = 0;
  // Binary search: each step keeps the half of the window that contains the wanted set bit.
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int __half_width = __digits / 2; __half_width >= 1; __half_width /= 2)
  {
    // __half_width < __digits, so neither shift reaches the width of the type.
    const auto __low_half_mask = static_cast<_Tp>((_Tp{1} << __half_width) - _Tp{1});
    const int __low_half_count = ::cuda::std::popcount(static_cast<_Tp>(__window & __low_half_mask));
    if (__remaining >= __low_half_count)
    {
      __remaining -= __low_half_count;
      __position += __half_width;
      __window = __window >> __half_width;
    }
  }
  return __position;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_FNS_H
