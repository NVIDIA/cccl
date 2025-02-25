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
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/cstdint>

#if __cccl_ptx_isa >= 600

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
struct WarpShuffleResult
{
  _Tp data;
  bool pred;

  template <typename _Up = _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE
  operator cuda::std::enable_if_t<!cuda::std::is_array_v<_Up>, _Up>() const
  {
    return data;
  }
};

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_idx(
  const _Tp& __data, int __src_lane, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size   = 32;
  constexpr int __is_void_ptr = _CUDA_VSTD::is_same_v<_Tp, void*> || _CUDA_VSTD::is_same_v<_Tp, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Tp> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
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
    ::memcpy(static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Tp));
#  pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_idx(__array[i], __pred, __src_lane, __clamp_segmask, __lane_mask);
    }
    WarpShuffleResult<_Tp> __result;
    __result.pred = __pred;
    ::memcpy(static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Tp));
    return __result;
  }
}

template <int _Width, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp>
warp_shuffle_idx(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return ::cuda::warp_shuffle_idx(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_up(
  const _Tp& __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size   = 32;
  constexpr int __is_void_ptr = _CUDA_VSTD::is_same_v<_Tp, void*> || _CUDA_VSTD::is_same_v<_Tp, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Tp> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
#  if __CUDA_ARCH__ >= 700
  [[maybe_unused]] int __pred;
  _CCCL_ASSERT(__match_all_sync(__activemask(), __delta, &__pred), "all active lanes must have the same delta");
#  endif
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__delta == 0, "delta must be 0 when Width == 1");
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (__warp_size - _Width) << 8;
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Tp));
#  pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_up(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    WarpShuffleResult<_Tp> __result;
    __result.pred = __pred;
    ::memcpy(static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Tp));
    return __result;
  }
}

template <int _Width, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp>
warp_shuffle_up(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return ::cuda::warp_shuffle_up(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_down(
  const _Tp& __data, int __delta, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size   = 32;
  constexpr int __is_void_ptr = _CUDA_VSTD::is_same_v<_Tp, void*> || _CUDA_VSTD::is_same_v<_Tp, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Tp> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
#  if __CUDA_ARCH__ >= 700
  [[maybe_unused]] int __pred;
  _CCCL_ASSERT(__match_all_sync(__activemask(), __delta, &__pred), "all active lanes must have the same delta");
#  endif
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__delta == 0, "delta must be 0 when Width == 1");
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__delta >= 1 && __delta < _Width, "delta must be in the range [1, _Width)");
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Tp));
#  pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_down(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    WarpShuffleResult<_Tp> __result;
    __result.pred = __pred;
    ::memcpy(static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Tp));
    return __result;
  }
}

template <int _Width, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp>
warp_shuffle_down(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return ::cuda::warp_shuffle_down(__data, __src_lane, 0xFFFFFFFF, __width);
}

template <int _Width = 32, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp> warp_shuffle_xor(
  const _Tp& __data, int __xor_mask, uint32_t __lane_mask = 0xFFFFFFFF, _CUDA_VSTD::integral_constant<int, _Width> = {})
{
  constexpr int __warp_size   = 32;
  constexpr int __is_void_ptr = _CUDA_VSTD::is_same_v<_Tp, void*> || _CUDA_VSTD::is_same_v<_Tp, const void*>;
  static_assert(!_CUDA_VSTD::is_pointer_v<_Tp> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(_CUDA_VSTD::has_single_bit(static_cast<uint32_t>(_Width)) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
#  if __CUDA_ARCH__ >= 700
  [[maybe_unused]] int __pred;
  _CCCL_ASSERT(__match_all_sync(__activemask(), __xor_mask, &__pred), "all active lanes must have the same delta");
#  endif
  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__xor_mask == 0, "delta must be 0 when Width == 1");
    return WarpShuffleResult<_Tp>{__data, true};
  }
  else
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(uint32_t));
    auto __clamp_segmask   = (_Width - 1) | ((__warp_size - _Width) << 8);
    bool __pred;
    uint32_t __array[__ratio];
    ::memcpy(static_cast<void*>(__array), static_cast<const void*>(_CUDA_VSTD::addressof(__data)), sizeof(_Tp));
#  pragma unroll
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = _CUDA_VPTX::shfl_sync_bfly(__array[i], __pred, __xor_mask, __clamp_segmask, __lane_mask);
    }
    WarpShuffleResult<_Tp> __result;
    __result.pred = __pred;
    ::memcpy(static_cast<void*>(_CUDA_VSTD::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Tp));
    return __result;
  }
}

template <int _Width, typename _Tp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE WarpShuffleResult<_Tp>
warp_shuffle_xor(const _Tp& __data, int __src_lane, _CUDA_VSTD::integral_constant<int, _Width> __width)
{
  return ::cuda::warp_shuffle_xor(__data, __src_lane, 0xFFFFFFFF, __width);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // __cccl_ptx_isa >= 600
#endif // _CUDA___WARP_COMMUNICATION_SHFL_H
