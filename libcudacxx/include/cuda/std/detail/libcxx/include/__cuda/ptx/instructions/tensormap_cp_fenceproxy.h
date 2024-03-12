// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_TENSORMAP_CP_FENCEPROXY_H_
#define _CUDA_PTX_TENSORMAP_CP_FENCEPROXY_H_

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <nv/target> // __CUDA_MINIMUM_ARCH__ and friends

#include "../ptx_dot_variants.h"
#include "../ptx_helper_functions.h"
#include "../../../cstdint"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

// 9.7.12.15.18. Parallel Synchronization and Communication Instructions: tensormap.cp_fenceproxy
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy
/*
// tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.sem.scope.sync.aligned  [dst], [src], size; // PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void tensormap_cp_fenceproxy(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  void* dst,
  const void* src,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
template <int _N32, dot_scope _Scope>
_CCCL_DEVICE static inline void tensormap_cp_fenceproxy(
  sem_release_t,
  scope_t<_Scope> __scope,
  void* __dst,
  const void* __src,
  n32_t<_N32> __size)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned  [%0], [%1], %2;"
        :
        : "l"(__as_ptr_gmem(__dst)),
          "r"(__as_ptr_smem(__src)),
          "n"(__size)
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_tensormap_cp_fenceproxy_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_TENSORMAP_CP_FENCEPROXY_H_
