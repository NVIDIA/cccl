//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_SHFL_SYNC_H
#define _CUDA_PTX_SHFL_SYNC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/std/__type_traits/is_integral.h>
#  include <cuda/std/__type_traits/is_signed.h>
#  include <cuda/std/cstdint>

#  include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

#  if __cccl_ptx_isa >= 600

template <dot_shfl_mode _ShuffleMode>
_CCCL_DEVICE static inline _CUDA_VSTD::uint32_t __shfl_sync_dst_lane(
  shfl_mode_t<_ShuffleMode> __shfl_mode,
  _CUDA_VSTD::uint32_t __lane_idx_offset,
  _CUDA_VSTD::uint32_t __clamp_segmask,
  _CUDA_VSTD::uint32_t __lane_mask)
{
  auto __lane     = get_sreg_laneid();
  auto __clamp    = __clamp_segmask & 0b11111;
  auto __segmask  = __clamp_segmask >> 8;
  auto __max_lane = (__lane & __segmask) | (__clamp & ~__segmask);
  auto __j        = 0;
  if constexpr (__shfl_mode == shfl_mode_idx)
  {
    auto __min_lane = (__lane & __clamp);
    __j             = __min_lane | (__lane_idx_offset & ~__segmask);
  }
  else if constexpr (__shfl_mode == shfl_mode_up)
  {
    __j = __lane - __lane_idx_offset;
  }
  else if constexpr (__shfl_mode == shfl_mode_down)
  {
    __j = __lane + __lane_idx_offset;
  }
  else
  {
    __j = __lane ^ __lane_idx_offset;
  }
  auto __dst = (__shfl_mode == shfl_mode_up) ? (__j >= __max_lane ? __j : __lane) : (__j <= __max_lane ? __j : __lane);
  return (1 << __dst);
}

struct shfl_return_values
{
  _CUDA_VSTD::uint32_t data;
  bool pred;
};

template <dot_shfl_mode _ShuffleMode>
_CCCL_NODISCARD _CCCL_DEVICE static inline shfl_return_values shfl_sync(
  shfl_mode_t<_ShuffleMode> __shfl_mode,
  _CUDA_VSTD::uint32_t __data,
  _CUDA_VSTD::uint32_t __lane_idx_offset,
  _CUDA_VSTD::uint32_t __clamp_segmask,
  _CUDA_VSTD::uint32_t __lane_mask) noexcept
{
  _CCCL_ASSERT(__lane_idx_offset < 32, "the lane index or offset must be less than the warp size");
  _CCCL_ASSERT((__clamp_segmask | 0b1111100011111) == 0b1111100011111,
               "clamp value + segmentation mask must be less or equal than 12 bits");
  _CCCL_ASSERT((__lane_mask & __activemask()) == __lane_mask, "lane mask must be a subset of the active mask");
  _CCCL_ASSERT(__shfl_sync_dst_lane(__shfl_mode, __lane_idx_offset, __clamp_segmask, __lane_mask) & __lane_mask,
               "the destination lane must be a member of the lane mask");
  _CUDA_VSTD::int32_t __pred;
  _CUDA_VSTD::uint32_t __ret;
  if constexpr (__shfl_mode == shfl_mode_idx)
  {
    asm volatile(
      "{                                                           \n\t\t"
      ".reg .pred p;                                               \n\t\t"
      "shfl_sync.sync.idx.b32 %0|p, %2, %3, %4, %5;                     \n\t\t"
      "selp.s32 %1, 1, 0, p;                                         \n\t"
      "}"
      : "=r"(__ret), "=r"(__pred)
      : "r"(__data), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  }
  else if constexpr (__shfl_mode == shfl_mode_up)
  {
    asm volatile(
      "{                                                           \n\t\t"
      ".reg .pred p;                                               \n\t\t"
      "shfl_sync.sync.up.b32 %0|p, %2, %3, %4, %5;                      \n\t\t"
      "selp.s32 %1, 1, 0, p;                                         \n\t"
      "}"
      : "=r"(__ret), "=r"(__pred)
      : "r"(__data), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  }
  else if constexpr (__shfl_mode == shfl_mode_down)
  {
    asm volatile(
      "{                                                           \n\t\t"
      ".reg .pred p;                                               \n\t\t"
      "shfl_sync.sync.down.b32 %0|p, %2, %3, %4, %5;                    \n\t\t"
      "selp.s32 %1, 1, 0, p;                                         \n\t"
      "}"
      : "=r"(__ret), "=r"(__pred)
      : "r"(__data), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  }
  else
  {
    asm volatile(
      "{                                                           \n\t\t"
      ".reg .pred p;                                               \n\t\t"
      "shfl_sync.sync.bfly.b32 %0|p, %2, %3, %4, %5;                    \n\t\t"
      "selp.s32 %1, 1, 0, p;                                         \n\t"
      "}"
      : "=r"(__ret), "=r"(__pred)
      : "r"(__data), "r"(__lane_idx_offset), "r"(__clamp_segmask), "r"(__lane_mask));
  }
  return shfl_return_values{__ret, static_cast<bool>(__pred)};
}

#  endif // __cccl_ptx_isa >= 600

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CCCL_STD_VER >= 2017
#endif // _CUDA_PTX_SHFL_SYNC_H
