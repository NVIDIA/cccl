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

#include "cuda/__cccl_config"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__ptx/instructions/shfl_sync.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
struct WarpShuffleResult
{
  _Tp data;
  bool pred;

  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE operator _Tp() const
  {
    return data;
  }
};

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle(
  _Tp __data, int __src_lane, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size = 32;
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  if constexpr (_Width == 1)
  {
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<void*>(&__data), sizeof(_Tp));
#pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_idx(__array[i], __pred, __src_lane, __clamp_segmask, __lane_mask);
    }
    _Tp __result;
    ::memcpy(static_cast<void*>(&__result), static_cast<void*>(__array), sizeof(_Tp));
    return WarpShuffleResult<_Tp>{__result, __pred};
  }
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_up(
  _Tp __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size = 32;
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
  if constexpr (_Width == 1)
  {
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<void*>(&__data), sizeof(_Tp));
#pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_up(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    _Tp __result;
    ::memcpy(static_cast<void*>(&__result), static_cast<void*>(__array), sizeof(_Tp));
    return WarpShuffleResult<_Tp>{__result, __pred};
  }
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_down(
  _Tp __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size = 32;
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
  if constexpr (_Width == 1)
  {
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<void*>(&__data), sizeof(_Tp));
#pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_down(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    _Tp __result;
    ::memcpy(static_cast<void*>(&__result), static_cast<void*>(__array), sizeof(_Tp));
    return WarpShuffleResult<_Tp>{__result, __pred};
  }
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_xor(
  _Tp __data, int __xor_mask, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size = 32;
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
  if constexpr (_Width == 1)
  {
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<void*>(&__data), sizeof(_Tp));
#pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_bfly(__array[i], __pred, __xor_mask, __clamp_segmask, __lane_mask);
    }
    _Tp __result;
    ::memcpy(static_cast<void*>(&__result), static_cast<void*>(__array), sizeof(_Tp));
    return WarpShuffleResult<_Tp>{__result, __pred};
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___WARP_COMMUNICATION_SHFL_H
