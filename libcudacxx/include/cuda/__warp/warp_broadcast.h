//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_WARP_BROADCAST_H
#define _CUDA___WARP_WARP_BROADCAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__warp/lane_mask.h>
#  include <cuda/__warp/warp_shuffle.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

template <class _Tp>
[[nodiscard]] _CCCL_DEVICE_API _Tp
warp_broadcast(const _Tp& __value, ::cuda::std::uint32_t __src_lane, lane_mask __lane_mask = lane_mask::all())
{
  return ::cuda::device::warp_shuffle_idx(__value, static_cast<int>(__src_lane), __lane_mask.value()).data;
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___WARP_WARP_BROADCAST_H
