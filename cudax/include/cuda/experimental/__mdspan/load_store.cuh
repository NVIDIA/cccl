//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_CUDA_LOAD_STORE
#define __CUDAX_CUDA_LOAD_STORE

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/ptx>
#include <cuda/std/type_traits>

#include <cuda/experimental/__mdspan/properties.cuh>

#define _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, EVICT_POLICY, PREFETCH)           \
  switch (PREFETCH)                                                               \
  {                                                                               \
    case no_prefetch:                                                             \
      return ::cuda::ptx::ld_global_##LOAD_BEHAVIOR##EVICT_POLICY(ptr);           \
    case prefetch_64B:                                                            \
      return ::cuda::ptx::ld_global_##LOAD_BEHAVIOR##EVICT_POLICY##_L2_64B(ptr);  \
    case prefetch_128B:                                                           \
      return ::cuda::ptx::ld_global_##LOAD_BEHAVIOR##EVICT_POLICY##_L2_128B(ptr); \
    case prefetch_256B:                                                           \
      return ::cuda::ptx::ld_global_##LOAD_BEHAVIOR##EVICT_POLICY##_L2_256B(ptr); \
  }

#define _CUDAX_LOAD_ADD_EVICTION_POLICY(LOAD_BEHAVIOR, EVICTION_POLICY, PREFETCH) \
  switch (EVICTION_POLICY)                                                        \
  {                                                                               \
    case eviction_normal:                                                         \
      _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, eviction_normal, PREFETCH);         \
    case eviction_first:                                                          \
      _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, eviction_first, PREFETCH);          \
    case eviction_last:                                                           \
      _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, eviction_last, PREFETCH);           \
    case eviction_last_use:                                                       \
      _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, eviction_last_use, PREFETCH);       \
    case eviction_no_alloc:                                                       \
      _CUDAX_LOAD_ADD_PREFETCH(LOAD_BEHAVIOR, eviction_no_alloc, PREFETCH);       \
  }

namespace cuda::experimental
{

template <typename T, MemoryBehavior B, EvictionPolicyEnum E, PrefetchSpatialEnum P>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE T
load(const T* ptr,
     memory_behavior_t<B> load_behavior   = read_write,
     eviction_policy_t<E> eviction_policy = eviction_normal,
     prefetch_spatial_t<P> prefetch       = no_prefetch_spatial) noexcept
{
#if _CCCL_PTX_LD_ENABLED
  if constexpr (eviction_policy == eviction_none && prefetch == no_prefetch_spatial)
  {
    return *ptr; // do not skip NVVM
  }
  else
  {
    // clang-format off
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_70,
      (
        if constexpr (load_behavior == read_write)
        {
          _CUDAX_LOAD_ADD_EVICTION_POLICY(, eviction_policy, prefetch);
        }
        else
        {
            _CUDAX_LOAD_ADD_EVICTION_POLICY(nc, eviction_policy, prefetch);
        }),
      (_CCCL_ASSERT(false, "eviction policy and prefetch require SM 7.0 or higher");));
    // clang-format on
  }
#else
  return *ptr;
#endif
}

template <typename T, EvictionPolicyEnum E>
_CCCL_DEVICE _CCCL_FORCEINLINE void
store(T value, T* ptr, eviction_policy_t<E> eviction_policy = eviction_normal) noexcept
{
  static_assert(!::cuda::std::is_const_v<T>);
#if _CCCL_PTX_ST_ENABLED
  if (eviction_policy == eviction_none)
  {
    *ptr = value;
  }
  else
  {
    // clang-format off
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
        (switch (eviction_policy)
         {
           case eviction_first:
             ::cuda::ptx::st_global_eviction_first(ptr, value);
           case eviction_last:
             ::cuda::ptx::st_global_eviction_last(ptr, value);
           case eviction_last_use:
             ::cuda::ptx::st_global_eviction_last_use(ptr, value);
           case eviction_no_alloc:
             ::cuda::ptx::st_global_eviction_no_alloc(ptr, value);
           default:
             _CCCL_UNREACHABLE();
         }),
        (_CCCL_ASSERT(false, "eviction policy requires SM 7.0 or higher");));
    // clang-format on
  }
#else
  *ptr = value;
#endif
}

} // namespace cuda::experimental

#undef _CUDAX_LOAD_ADD_PREFETCH
#undef _CUDAX_LOAD_ADD_EVICTION_POLICY

#endif // __CUDAX_CUDA_LOAD_STORE
