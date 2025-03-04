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

#    include <cuda/__data_movement/properties.h>
#    include <cuda/__ptx/instructions/ld.h>
#    include <cuda/std/__bit/bit_cast.h>

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
  _CCCL_ASSERT(__ptr != nullptr, "cuda::load: 'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "cuda::load: 'ptr' must point to global memory");
  if constexpr (__memory_access == read_write && __eviction_policy == eviction_none
                && __prefetch == prefetch_spatial_none)
  {
    return *__ptr;
  }
  else
  {
    static_assert(sizeof(_Tp) <= 16, "cuda::load with non-default properties only supports types up to 16 bytes");
    return ::cuda::device::__load(__ptr, __memory_access, __eviction_policy, __prefetch);
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  endif // __cccl_ptx_isa >= 700
#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
