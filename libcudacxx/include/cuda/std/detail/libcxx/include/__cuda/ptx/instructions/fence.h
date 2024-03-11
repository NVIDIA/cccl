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

#ifndef _CUDA_PTX_FENCE_H_
#define _CUDA_PTX_FENCE_H_

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

// 9.7.12.4. Parallel Synchronization and Communication Instructions: membar/fence
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence
/*
// fence{.sem}.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .sc, .acq_rel }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 600
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_70__();
template <dot_sem _Sem, dot_scope _Scope>
_CCCL_DEVICE static inline void fence(
  sem_t<_Sem> __sem,
  scope_t<_Scope> __scope)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  static_assert(__scope == scope_cta || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_cta) {
      asm volatile (
        "fence.sc.cta; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_gpu) {
      asm volatile (
        "fence.sc.gpu; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc && __scope == scope_sys) {
      asm volatile (
        "fence.sc.sys; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_cta) {
      asm volatile (
        "fence.acq_rel.cta; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_gpu) {
      asm volatile (
        "fence.acq_rel.gpu; // 1."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel && __scope == scope_sys) {
      asm volatile (
        "fence.acq_rel.sys; // 1."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_is_not_supported_before_SM_70__();
  ));
}
#endif // __cccl_ptx_isa >= 600

/*
// fence{.sem}.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .sc, .acq_rel }
// .scope     = { .cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <dot_sem _Sem>
_CCCL_DEVICE static inline void fence(
  sem_t<_Sem> __sem,
  scope_cluster_t)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  // __scope == scope_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_sc) {
      asm volatile (
        "fence.sc.cluster; // 2."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__sem == sem_acq_rel) {
      asm volatile (
        "fence.acq_rel.cluster; // 2."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780
/*
// fence.mbarrier_init.sem.scope; // 3. PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
template <typename=void>
__device__ static inline void fence_mbarrier_init(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void fence_mbarrier_init(
  sem_release_t,
  scope_cluster_t)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "fence.mbarrier_init.release.cluster; // 3."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_mbarrier_init_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// fence.proxy.alias; // 4. PTX ISA 75, SM_70
template <typename=void>
__device__ static inline void fence_proxy_alias();
*/
#if __cccl_ptx_isa >= 750
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
template <typename=void>
_CCCL_DEVICE static inline void fence_proxy_alias()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70,(
    asm volatile (
      "fence.proxy.alias; // 4."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_alias_is_not_supported_before_SM_70__();
  ));
}
#endif // __cccl_ptx_isa >= 750
/*
// fence.proxy.async; // 5. PTX ISA 80, SM_90
template <typename=void>
__device__ static inline void fence_proxy_async();
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void fence_proxy_async()
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm volatile (
      "fence.proxy.async; // 5."
      :
      :
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
// .space     = { .global, .shared::cluster, .shared::cta }
template <cuda::ptx::dot_space Space>
__device__ static inline void fence_proxy_async(
  cuda::ptx::space_t<Space> space);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
template <dot_space _Space>
_CCCL_DEVICE static inline void fence_proxy_async(
  space_t<_Space> __space)
{
  static_assert(__space == space_global || __space == space_cluster || __space == space_shared, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_global) {
      asm volatile (
        "fence.proxy.async.global; // 6."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_cluster) {
      asm volatile (
        "fence.proxy.async.shared::cluster; // 6."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__space == space_shared) {
      asm volatile (
        "fence.proxy.async.shared::cta; // 6."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_async_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// fence.proxy.tensormap::generic.release.scope; // 7. PTX ISA 83, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_tensormap_generic(
  sem_release_t,
  scope_t<_Scope> __scope)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.cta; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.cluster; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.gpu; // 7."
        :
        :
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "fence.proxy.tensormap::generic.release.sys; // 7."
        :
        :
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830

/*
// fence.proxy.tensormap::generic.sem.scope [addr], size; // 8. PTX ISA 83, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster, .gpu, .sys }
template <int N32, cuda::ptx::dot_scope Scope>
__device__ static inline void fence_proxy_tensormap_generic(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  const void* addr,
  cuda::ptx::n32_t<N32> size);
*/
#if __cccl_ptx_isa >= 830
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
template <int _N32, dot_scope _Scope>
_CCCL_DEVICE static inline void fence_proxy_tensormap_generic(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  const void* __addr,
  n32_t<_N32> __size)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.cta [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.cluster [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_gpu) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.gpu [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_sys) {
      asm volatile (
        "fence.proxy.tensormap::generic.acquire.sys [%0], %1; // 8."
        :
        : "l"(__addr),
          "n"(__size)
        : "memory"
      );
    }
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_fence_proxy_tensormap_generic_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 830

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_FENCE_H_
