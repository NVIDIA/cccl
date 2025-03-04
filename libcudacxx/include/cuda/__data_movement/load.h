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

#  include <cuda/__data_movement/properties.h>
#  include <cuda/__ptx/instructions/ld.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

#  if 1

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

#  else
#    define _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, EVICT_POLICY, PREFETCH) \
      if constexpr ((PREFETCH) == prefetch_spatial_none)                   \
      {                                                                    \
        return _CUDA_VPTX::ld_global##LOAD_BEHAVIOR##EVICT_POLICY(__ptr);  \
      }

#    define _CCCL_LOAD_ADD_EVICTION_POLICY(LOAD_BEHAVIOR, EVICTION_POLICY, PREFETCH) \
      if constexpr ((EVICTION_POLICY) == eviction_none)                              \
      {                                                                              \
        _CCCL_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, "", PREFETCH);                        \
      }
#  endif

template <typename _Tp, _MemoryBehavior _Bp, _EvictionPolicyEnum _Ep, _PrefetchSpatialEnum _Pp>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp __load_sm70(
  const _Tp* __ptr,
  __memory_behavior_t<_Bp> __load_behavior,
  __eviction_policy_t<_Ep> __eviction_policy,
  __prefetch_spatial_t<_Pp> __prefetch) noexcept
{
#  if __libcuda_ptx_isa >= 700
  // TODO: cvta.to.global
  if constexpr (__load_behavior == read_write)
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(, __eviction_policy, __prefetch);
  }
  else
  {
    _CCCL_LOAD_ADD_EVICTION_POLICY(_nc, __eviction_policy, __prefetch);
  }
#  endif
}

#  undef _CCCL_LOAD_ADD_PREFETCH
#  undef _CCCL_LOAD_ADD_EVICTION_POLICY

_CCCL_DEVICE void __eviction_policy_and_prefetch_require_SM_70_or_higher();

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp,
          _MemoryBehavior _Bp      = _MemoryBehavior::_ReadWrite,
          _EvictionPolicyEnum _Ep  = _EvictionPolicyEnum::_None,
          _PrefetchSpatialEnum _Pp = _PrefetchSpatialEnum::_None>
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _Tp
load(const _Tp* __ptr,
     __memory_behavior_t<_Bp> __load_behavior   = read_write,
     __eviction_policy_t<_Ep> __eviction_policy = eviction_none,
     __prefetch_spatial_t<_Pp> __prefetch       = prefetch_spatial_none) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "cuda::load: 'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "cuda::load: 'ptr' must point to global memory");
  if constexpr (__load_behavior == read_write && __eviction_policy == eviction_none
                && __prefetch == prefetch_spatial_none)
  {
    return *__ptr;
  }
  else
  {
    static_assert(sizeof(_Tp) <= 16, "cuda::load with non-default properties only supports types up to 16 bytes");
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                      (return ::cuda::__load_sm70(__ptr, __load_behavior, __eviction_policy, __prefetch);),
                      (__eviction_policy_and_prefetch_require_SM_70_or_higher();));
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_LOAD_H
