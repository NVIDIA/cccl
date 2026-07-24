//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___BIT_BIT_REPEAT_H
#define _CUDA_STD___BIT_BIT_REPEAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/num_bits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp bit_repeat(_Tp __v, int __n) noexcept(false)
{
  // If __n <= 0, the function must fail during constant evaluation.
  _CCCL_IF_CONSTEVAL_DEFAULT
  {
    _CCCL_VERIFY(__n > 0, "__n must be greater than 0");
  }
  else
  {
    _CCCL_ASSERT(__n > 0, "__n must be greater than 0");
  }

  constexpr int __width = __num_bits_v<_Tp>;
  if (__n >= __width)
  {
    return __v;
  }

  const auto __pattern = static_cast<_Tp>(__v & ::cuda::bitmask<_Tp>(0, __n));
  _Tp __ret            = __pattern;
  for (int __i = __n; __i < __width; __i += __n)
  {
    __ret <<= __n;
    __ret |= __pattern;
  }
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___BIT_BIT_REPEAT_H
