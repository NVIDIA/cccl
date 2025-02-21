//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPO__RATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_COMMUNICATION_SHFL_H
#define _CUDA___WARP_COMMUNICATION_SHFL_H

#include <cuda/std/detail/__config>

#include <cstdint>

#include "cuda/__cmath/ceil_div.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/shfl_sync.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
shfl(_Tp __data, int __src_lane, uint32_t __mask = 0xFFFFFFFF, int __width = /*warp size=*/32)
{
  _CCCL_ASSERT(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(__width)) && __width <= 32,
               "__width must be a power of 2 and less or equal to the warp size");
  if (__width == 1)
  {
    return __data;
  }
  constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
  uint32_t __array[__ratio];
  ::memcpy(__array, &__data, sizeof(_Tp));
#pragma unroll
  for (int i = 0; i < __ratio; ++i)
  {
    __array[i] = ::cuda::ptx::shfl(__array[i], __src_lane, __mask, __width);
  }
  _Tp result;
  ::memcpy(&result, __array, sizeof(_Tp));
  return result;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___WARP_COMMUNICATION_SHFL_H
