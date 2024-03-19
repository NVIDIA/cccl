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

#ifndef _CUDA_PTX_MBARRIER_WAIT_H_
#define _CUDA_PTX_MBARRIER_WAIT_H_

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

// 9.7.12.15.16. Parallel Synchronization and Communication Instructions: mbarrier.test_wait/mbarrier.try_wait
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait
/*
// mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait(
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_80__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_test_wait(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.test_wait.shared.b64 P_OUT, [%1], %2;                                                  // 1. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "l"(__state)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_80__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2.   PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_test_wait(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.acquire.cta.shared::cta.b64        P_OUT, [%1], %2;                        // 2.  \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.acquire.cluster.shared::cta.b64        P_OUT, [%1], %2;                        // 2.  \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity;                                     // 3.  PTX ISA 71, SM_80
template <typename=void>
__device__ static inline bool mbarrier_test_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 710
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_80__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_test_wait_parity(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.test_wait.parity.shared.b64 P_OUT, [%1], %2;                                     // 3. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__phaseParity)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_80__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 710

/*
// mbarrier.test_wait.parity{.sem}{.scope}.shared::cta.b64 waitComplete, [addr], phaseParity;                  // 4.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_test_wait_parity(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 P_OUT, [%1], %2;                  // 4. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 P_OUT, [%1], %2;                  // 4. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_test_wait_parity_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state;                                      // 5a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.shared::cta.b64         P_OUT, [%1], %2;                                      // 5a. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "l"(__state)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    // 5b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait(
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state,
  const _CUDA_VSTD::uint32_t& __suspendTimeHint)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.shared::cta.b64         P_OUT, [%1], %2, %3;                    // 5b. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "l"(__state),
        "r"(__suspendTimeHint)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state;                        // 6a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait{.sem}{.scope}.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      // 6b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint64_t& __state,
  const _CUDA_VSTD::uint32_t& __suspendTimeHint)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64         P_OUT, [%1], %2 , %3;      // 6b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state),
          "r"(__suspendTimeHint)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2 , %3;      // 6b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "l"(__state),
          "r"(__suspendTimeHint)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800
/*
// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity;                                // 7a.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2;                                // 7a. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__phaseParity)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint;               // 7b.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline bool mbarrier_try_wait_parity(
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity,
  const _CUDA_VSTD::uint32_t& __suspendTimeHint)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    asm (
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2, %3;               // 7b. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__phaseParity),
        "r"(__suspendTimeHint)
      : "memory"
    );
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity;                  // 8a.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  P_OUT, [%1], %2;                  // 8a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  P_OUT, [%1], %2;                  // 8a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.try_wait.parity{.sem}{.scope}.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 8b.  PTX ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_try_wait_parity(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint32_t& phaseParity,
  const uint32_t& suspendTimeHint);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_try_wait_parity(
  sem_acquire_t,
  scope_t<_Scope> __scope,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __phaseParity,
  const _CUDA_VSTD::uint32_t& __suspendTimeHint)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint32_t __waitComplete;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  P_OUT, [%1], %2, %3; // 8b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity),
          "r"(__suspendTimeHint)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  P_OUT, [%1], %2, %3; // 8b. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__phaseParity),
          "r"(__suspendTimeHint)
        : "memory"
      );
    }
    return static_cast<bool>(__waitComplete);
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_try_wait_parity_is_not_supported_before_SM_90__();
    return false;
  ));
}
#endif // __cccl_ptx_isa >= 800

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_MBARRIER_WAIT_H_
