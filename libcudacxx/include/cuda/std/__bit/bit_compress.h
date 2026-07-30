//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_COMPRESS_H
#define _CUDA_STD___BIT_BIT_COMPRESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/std/__bit/bit_reverse.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_compress(_Tp __v, const _Tp __mask) noexcept
{
  if (__mask == static_cast<_Tp>(~_Tp{0}))
  {
    return __v;
  }

  // Work on reversed mask, so we can use __builtin_clz.
  auto __mask_rev = ::cuda::std::bit_reverse(__mask);

  _Tp __ret{0};
  int __offset = 0;

  for (auto __skip = ::cuda::std::countl_zero(__mask_rev); __skip != __num_bits_v<_Tp>;
       __skip      = ::cuda::std::countl_zero(__mask_rev))
  {
    // Skip leading zeros in the mask.
    __mask_rev <<= __skip;
    __v >>= __skip;

    // Find out how many consecutive bits we can write.
    const auto __n = ::cuda::std::countl_one(__mask_rev);

    // Write __n consecutive bits.
    const auto __segment = static_cast<_Tp>(__v & ::cuda::bitmask<_Tp>(0, __n));
    __ret                = static_cast<_Tp>(__ret | static_cast<_Tp>(__segment << __offset));
    __offset += __n;

    // Remove written bits from __v and __mask_rev.
    __mask_rev <<= __n;
    __v >>= __n;
  }
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_COMPRESS_H
