// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_PTX_PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION_INSTRUCTIONS_MBARRIER_H_
#define _CUDA_PTX_PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION_INSTRUCTIONS_MBARRIER_H_

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

#include "ptx_dot_variants.h"
#include "ptx_helper_functions.h"
#include "../../cstdint"

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX

/*
 *  9.7.12.15. Parallel Synchronization and Communication Instructions: mbarrier
 *   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier
 *
 */

// 9.7.12.15.9. Parallel Synchronization and Communication Instructions: mbarrier.init
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init

// 9.7.12.15.10. Parallel Synchronization and Communication Instructions: mbarrier.inval
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval

// 9.7.12.15.11. Parallel Synchronization and Communication Instructions: mbarrier.expect_tx
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx

// 9.7.12.15.12. Parallel Synchronization and Communication Instructions: mbarrier.complete_tx
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx

// 9.7.12.15.13. Parallel Synchronization and Communication Instructions: mbarrier.arrive
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive

/*
PTX ISA docs:

// mbarrier.arrive:
mbarrier.arrive{.shared}.b64 state, [addr];                                         // 1. PTX ISA 70, SM_80
mbarrier.arrive{.shared{::cta}}.b64 state, [addr]{, count};                         // 2. PTX ISA 78, SM_90 (due to count)

mbarrier.arrive{.sem}{.scope}{.shared{::cta}}.b64           state, [addr]{, count}; // 3. PTX ISA 80, SM_90 (some variants are SM_80, but are covered by 1)
mbarrier.arrive{.sem}{.scope}{.shared::cluster}.b64         _, [addr] {,count}      // 4. PTX ISA 80, SM_90

.sem   = { .release }
.scope = { .cta, .cluster }


// mbarrier.arrive.noComplete:
mbarrier.arrive.noComplete{.shared}.b64 state, [addr], count;                       // 5. PTX ISA 70, SM_80
mbarrier.arrive.noComplete{.shared{::cta}}.b64 state, [addr], count;                // 6. PTX ISA 78, Not exposed. Just a spelling change (shared -> shared::cta)
mbarrier.arrive.noComplete{.sem}{.cta}{.shared{::cta}}.b64  state, [addr], count;   // 7. PTX ISA 80, Not exposed. Adds .release, and .cta scope.


// mbarrier.arrive.expect_tx:
mbarrier.arrive.expect_tx{.sem}{.scope}{.shared{::cta}}.b64 state, [addr], tx_count; // 8. PTX ISA 80, SM_90
mbarrier.arrive.expect_tx{.sem}{.scope}{.shared::cluster}.b64   _, [addr], tx_count; // 9. PTX ISA 80, SM_90

.sem   = { .release }
.scope = { .cta, .cluster }


Corresponding Exposure:

// mbarrier_arrive:
mbarrier.arrive.shared.b64                                  state,  [addr];           // 1. PTX ISA 70, SM_80, !memory
// count is non-optional, otherwise 3 would not be distinguishable from 1
mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2. PTX ISA 78, SM_90, !memory
mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr];           // 3a. PTX ISA 80, SM_90, !memory
.space = { .shared::cta}
.sem   = { .release }
.scope = { .cta, .cluster }

mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr], count;    // 3b. PTX ISA 80, SM_90, !memory
.space = { .shared::cta}
.sem   = { .release }
.scope = { .cta, .cluster }

// NOTE: .scope=.cta is dropped on purpose
mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr];                // 4a. PTX ISA 80, SM_90, !memory
.space = { .shared::cluster}
.sem   = { .release }
.scope = { .cluster }

// NOTE: .scope=.cta is dropped on purpose
mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr], count;         // 4b. PTX ISA 80, SM_90, !memory
.space = { .shared::cluster}
.sem   = { .release }
.scope = { .cluster }


// mbarrier_arrive_no_complete:
mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5. PTX ISA 70, SM_80, !memory


mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64        state, [addr], tx_count;   // 8. PTX ISA 80, SM_90, !memory
.space = { .shared::cta }
.sem   = { .release }
.scope = { .cta, .cluster }

// NOTE: .scope=.cta is dropped on purpose
mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64        _, [addr], tx_count;       // 9. PTX ISA 80, SM_90, !memory
.space = { .shared::cluster }
.sem   = { .release }
.scope = { .cluster }

*/

/*
// mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  _CUDA_VSTD::uint64_t* __addr)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    _CUDA_VSTD::uint64_t __state;
    asm (
      "mbarrier.arrive.shared.b64                                  %0,  [%1];           // 1. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr))
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.  PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint64_t __state;
    asm (
      "mbarrier.arrive.shared::cta.b64                             %0,  [%1], %2;    // 2. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr];           // 3a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  sem_release_t,
  scope_t<_Scope> __scope,
  space_shared_t,
  _CUDA_VSTD::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint64_t __state;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   state,  [addr], count;    // 3b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  sem_release_t,
  scope_t<_Scope> __scope,
  space_shared_t,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint64_t __state;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.release.cta.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr];                // 4a.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  sem_release_t,
  scope_cluster_t,
  space_cluster_t,
  _CUDA_VSTD::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0];                // 4a. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr))
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive{.sem}{.scope}{.space}.b64                   _, [addr], count;         // 4b.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void mbarrier_arrive(
  sem_release_t,
  scope_cluster_t,
  space_cluster_t,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [%0], %1;         // 4b. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)),
        "r"(__count)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
template <typename=void>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_no_complete(
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    _CUDA_VSTD::uint64_t __state;
    asm (
      "mbarrier.arrive.noComplete.shared.b64                       %0,  [%1], %2;    // 5. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64 state, [addr], tx_count; // 8.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cta, .cluster }
// .space     = { .shared::cta }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_expect_tx(
  sem_release_t,
  scope_t<_Scope> __scope,
  space_shared_t,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    _CUDA_VSTD::uint64_t __state;
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2; // 8. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 %0, [%1], %2; // 8. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
    return 0;
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.expect_tx{.sem}{.scope}{.space}.b64   _, [addr], tx_count; // 9.  PTX ISA 80, SM_90
// .sem       = { .release }
// .scope     = { .cluster }
// .space     = { .shared::cluster }
template <typename=void>
__device__ static inline void mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t,
  cuda::ptx::scope_cluster_t,
  cuda::ptx::space_cluster_t,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <typename=void>
_CCCL_DEVICE static inline void mbarrier_arrive_expect_tx(
  sem_release_t,
  scope_cluster_t,
  space_cluster_t,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64   _, [%0], %1; // 9. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)),
        "r"(__tx_count)
      : "memory"
    );
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

// 9.7.12.15.14. Parallel Synchronization and Communication Instructions: mbarrier.arrive_drop
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop

// 9.7.12.15.15. Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive

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

// 9.7.12.15.17. Parallel Synchronization and Communication Instructions: mbarrier.pending_count
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION_INSTRUCTIONS_MBARRIER_H_
