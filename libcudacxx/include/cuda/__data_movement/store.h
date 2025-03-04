//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_STORE_H
#define _CUDA___DATA_MOVEMENT_STORE_H

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
#    include <cuda/__ptx/instructions/st.h>
#    include <cuda/std/__type_traits/is_const.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp, _EvictionPolicyEnum _Ep>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
__store_sm70(_Tp __data, _Tp* __ptr, __eviction_policy_t<_Ep> __eviction_policy) noexcept
{
  // TODO: cvta.to.global
  if constexpr (__eviction_policy == eviction_unchanged)
  {
    _CUDA_VPTX::st_global_L1_evict_unchanged(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_normal)
  {
    _CUDA_VPTX::st_global_L1_evict_normal(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_first)
  {
    _CUDA_VPTX::st_global_L1_evict_first(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_last)
  {
    _CUDA_VPTX::st_global_L1_evict_last(__ptr, __data);
  }
  else if constexpr (__eviction_policy == eviction_no_alloc)
  {
    _CUDA_VPTX::st_global_L1_no_allocate(__ptr, __data);
  }
}

extern "C" _CCCL_DEVICE void __eviction_policy_require_SM_70_or_higher();

/***********************************************************************************************************************
 * USER API
 **********************************************************************************************************************/

template <typename _Tp, _EvictionPolicyEnum _Ep>
_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void
store(_Tp __data, _Tp* __ptr, __eviction_policy_t<_Ep> __eviction_policy = eviction_none) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "cuda::store: 'ptr' must not be null");
  _CCCL_ASSERT(__isGlobal(__ptr), "cuda::store: 'ptr' must point to global memory");
  static_assert(!_CUDA_VSTD::is_const_v<_Tp>);
  if constexpr (__eviction_policy == eviction_none)
  {
    *__ptr = __data;
  }
  else
  {
    static_assert(sizeof(_Tp) <= 16, "cuda::store with non-default properties only supports types up to 16 bytes");
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,
                      (::cuda::__store_sm70(__data, __ptr, __eviction_policy);),
                      (::cuda::__eviction_policy_require_SM_70_or_higher();));
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  endif // __cccl_ptx_isa >= 700
#endif // _CCCL_HAS_CUDA_COMPILER
#endif // _CUDA___DATA_MOVEMENT_STORE_H
