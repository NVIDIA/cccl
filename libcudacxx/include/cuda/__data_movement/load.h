//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_LOAD_H
#define _CUDA___DATA_MOVEMENT_LOAD_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER
#  if __cccl_ptx_isa >= 700

#    include <cuda/__barrier/aligned_size.h>
#    include <cuda/__data_movement/properties.h>
#    include <cuda/__ptx/instructions/ld.h>
#    include <cuda/std/__algorithm/min.h>
#    include <cuda/std/__bit/bit_cast.h>
#    include <cuda/std/__bit/has_single_bit.h>
#    include <cuda/std/__memory/assume_aligned.h>
#    include <cuda/std/__type_traits/integral_constant.h>
#    include <cuda/std/__utility/integer_sequence.h>
#    include <cuda/std/array>
#    include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

#    define _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, EVICT_POLICY, PREFETCH)          \
      if constexpr ((PREFETCH) == prefetch_spatial_none)                            \
      {                                                                             \
        return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY(__ptr);           \
      }                                                                             \
      else if constexpr ((PREFETCH) == prefetch_64B)                                \
      {                                                                             \
        return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_64B(__ptr);  \
      }                                                                             \
      else if constexpr ((PREFETCH) == prefetch_128B)                               \
      {                                                                             \
        return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_128B(__ptr); \
      }                                                                             \
      else if constexpr ((PREFETCH) == prefetch_256B)                               \
      {                                                                             \
        return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY##_L2_256B(__ptr); \
      }

#    define _CCCL_LOAD_ADD_EVICTION_POLICY(LOAD_BEHAVIOR, EVICTION_POLICY, PREFETCH) \
      if constexpr ((EVICTION_POLICY) == eviction_none)                              \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, , PREFETCH);                          \
      }                                                                              \
      else if constexpr ((EVICTION_POLICY) == eviction_normal)                       \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_normal, PREFETCH);          \
      }                                                                              \
      else if constexpr ((EVICTION_POLICY) == eviction_first)                        \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_first, PREFETCH);           \
      }                                                                              \
      else if constexpr ((EVICTION_POLICY) == eviction_last)                         \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_last, PREFETCH);            \
      }                                                                              \
      else if constexpr ((EVICTION_POLICY) == eviction_unchanged)                    \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_evict_unchanged, PREFETCH);       \
      }                                                                              \
      else if constexpr ((EVICTION_POLICY) == eviction_no_alloc)                     \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, _L1_no_allocate, PREFETCH);           \
      }

template <typename _Tp, _MemoryAccess _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load(
  const _Tp* __ptr1,
  __memory_access_t<_Bp> __memory_access,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch) noexcept
{
  static_assert(sizeof(_Tp) <= 16);
  auto __ptr = _CUDA_VSTD::bit_cast<const _Tp*>(__cvta_generic_to_global(__ptr1));
  if constexpr (__memory_access == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, __prefetch);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, __prefetch);
  }
}

#    undef _CCCL_LOAD_ADD_PREFETCH
#    undef _CCCL_LOAD_ADD_EVICTION_POLICY

/***********************************************************************************************************************
 * UTILITIES
 **********************************************************************************************************************/

template <typename _Op, size_t... _Ip>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void __unroll(const _Op& __op, _CUDA_VSTD::index_sequence<_Ip...> = {})
{
  (__op(_CUDA_VSTD::integral_constant<size_t, _Ip>{}), ...);
}

template <size_t _Bytes>
struct alignas(_Bytes) __DataNBytes
{
  uint8_t __tmp[_Bytes];
};

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(const _Tp* __ptr,
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "cuda::device::load: 'ptr' must not be null");
  _CCCL_ASSERT(reinterpret_cast<size_t>(__ptr) % alignof(_Tp) == 0, "cuda::device::load: 'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "cuda::device::load: 'ptr' must point to global memory");
  if constexpr (__memory_access == read_write && __eviction_policy == eviction_none
                && __prefetch == prefetch_spatial_none)
  {
    return *__ptr;
  }
  else
  {
    static_assert(_CUDA_VSTD::has_single_bit(sizeof(_Tp)), "'sizeof(_Tp)' must be a power of 2");
    static_assert(_CUDA_VSTD::has_single_bit(alignof(_Tp)), "'alignof(_Tp)' must be a power of 2");
    constexpr auto __num_bytes = alignof(_Tp);
    if constexpr (__num_bytes > 16)
    {
      constexpr auto __num_16_bytes = __num_bytes / 16;
      using __load_type             = _CUDA_VSTD::array<__DataNBytes<16>, __num_16_bytes>;
      auto __ptr1                   = reinterpret_cast<const __load_type*>(__ptr);
      __load_type __tmp;
      auto __lambda = [&](auto __i) { // TODO: handle nvrtc
        __tmp[__i] = ::cuda::device::__load(__ptr1[__i], __memory_access, __eviction_policy, __prefetch);
      };
      ::cuda::device::__unroll(__lambda, _CUDA_VSTD::make_index_sequence<__num_16_bytes>{});
      return _CUDA_VSTD::bit_cast<_Tp>(__tmp);
    }
    else
    {
      using __load_type = __DataNBytes<__num_bytes>;
      auto __ptr1       = reinterpret_cast<const __load_type*>(__ptr);
      auto __tmp        = ::cuda::device::__load(__ptr1, __memory_access, __eviction_policy, __prefetch);
      return _CUDA_VSTD::bit_cast<_Tp>(__tmp);
    }
  }
}

template <size_t _Np,
          typename _Tp,
          size_t _Align            = alignof(_Tp),
          _MemoryAccess _Bp        = _MemoryAccess::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CUDA_VSTD::array<_Tp, _Np>
load(const _Tp* __ptr,
     ::cuda::aligned_size_t<_Align>             = {},
     __memory_access_t<_Bp> __memory_access     = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  static_assert(_Align >= alignof(_Tp), "'ptr' must be aligned to at least 'alignof(_Tp)'");
  static_assert(sizeof(_Tp) * _Np % _Align == 0, "Np * sizeof(_Tp) must be a multiple of _Align");
  _CCCL_ASSERT(__ptr != nullptr, "cuda::device::load: 'ptr' must not be null");
  _CCCL_ASSERT(reinterpret_cast<size_t>(__ptr) % _Align == 0, "cuda::device::load: 'ptr' must be aligned");
  _CCCL_ASSERT(__isGlobal(__ptr), "cuda::device::load: 'ptr' must point to global memory");
  constexpr auto __count = sizeof(_Tp) * _Np / _Align;
  using __load_type      = __DataNBytes<_Align>;
  auto __ptr1            = reinterpret_cast<const __load_type*>(__ptr);
  _CUDA_VSTD::array<__load_type, __count> __tmp;
  auto __lambda = [&](auto __i) { // TODO: handle nvrtc
    __tmp[__i] = ::cuda::device::__load(__ptr1[__i], __memory_access, __eviction_policy, __prefetch);
  };
  ::cuda::device::__unroll(__lambda, _CUDA_VSTD::make_index_sequence<__count>{});
  using __result_t = _CUDA_VSTD::array<_Tp, _Np>;
  return _CUDA_VSTD::bit_cast<__result_t>(__tmp);
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  endif // __cccl_ptx_isa >= 700
#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
