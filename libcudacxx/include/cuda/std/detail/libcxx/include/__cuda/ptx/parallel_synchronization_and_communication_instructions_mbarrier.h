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

#include "ptx_dot_variants.h"
#include "ptx_helper_functions.h"
#include "ptx_isa_target_macros.h"
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

// mbarrier.arrive{.sem}{.scope}{.shared{::cta}}.b64           state, [addr]{, count};
// mbarrier.arrive{.sem}{.scope}{.shared::cluster}.b64         _, [addr] {,count}
// mbarrier.arrive.expect_tx{.sem}{.scope}{.shared{::cta}}.b64 state, [addr], txCount;
// mbarrier.arrive.expect_tx{.sem}{.scope}{.shared::cluster}.b64   _, [addr], txCount;
// mbarrier.arrive.noComplete{.sem}{.cta}{.shared{::cta}}.b64  state, [addr], count;
//
// .sem   = { .release }
// .scope = { .cta, .cluster }

/*
// mbarrier.arrive.shared.b64 state, [addr]; // 1.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_cta_t scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  sem_release_t __sem,
  scope_cta_t __scope,
  space_shared_t __space,
  _CUDA_VSTD::uint64_t* __addr)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cta (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)

  _CUDA_VSTD::uint64_t __state;

  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    asm (
      "mbarrier.arrive.shared.b64 %0, [%1]; // 1. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr))
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
  ));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.noComplete.shared.b64 state, [addr], count; // 2.  PTX ISA 70, SM_80
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_cta_t scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_no_complete(
  sem_release_t __sem,
  scope_cta_t __scope,
  space_shared_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cta (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)

  _CUDA_VSTD::uint64_t __state;

  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,(
    asm (
      "mbarrier.arrive.noComplete.shared.b64 %0, [%1], %2; // 2. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
  ));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.arrive.shared.b64 state, [addr], count; // 3. PTX ISA 78, SM_90
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_cta_t scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 780
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  sem_release_t __sem,
  scope_cta_t __scope,
  space_shared_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cta (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)

  _CUDA_VSTD::uint64_t __state;

  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "mbarrier.arrive.shared.b64 %0, [%1], %2; // 3."
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 780

/*
// mbarrier.arrive.release.cluster.shared.b64 state,  [addr], count; // 4. PTX ISA 80, SM_90
template <typename=void>
__device__ static inline uint64_t mbarrier_arrive(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_cluster_t scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
  sem_release_t __sem,
  scope_cluster_t __scope,
  space_shared_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  // __scope == scope_cluster (due to parameter type constraint)
  // __space == space_shared (due to parameter type constraint)

  _CUDA_VSTD::uint64_t __state;

  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    asm (
      "mbarrier.arrive.release.cluster.shared.b64 %0,  [%1], %2; // 4."
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.release{.scope}.shared::cluster.b64 _, [addr], count;   // 5.  PTX ISA 80, SM_90
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_arrive(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_cluster_t space,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 800
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline void mbarrier_arrive(
  sem_release_t __sem,
  scope_t<_Scope> __scope,
  space_shared_cluster_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared_cluster (due to parameter type constraint)



  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], %1;   // 5. "
        :
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0], %1;   // 5. "
        :
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    }

  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ void __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
    return __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.expect_tx.release{.scope}.shared.b64   state, [addr], tx_count; // 6.  PTX ISA 80, SM_90
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline uint64_t mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_t space,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_expect_tx(
  sem_release_t __sem,
  scope_t<_Scope> __scope,
  space_shared_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared (due to parameter type constraint)

  _CUDA_VSTD::uint64_t __state;

  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.expect_tx.release.cta.shared.b64   %0, [%1], %2; // 6. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.expect_tx.release.cluster.shared.b64   %0, [%1], %2; // 6. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800

/*
// mbarrier.arrive.expect_tx.release{.scope}.shared::cluster.b64 _, [addr], tx_count; // 7.  PTX ISA 80, SM_90
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline void mbarrier_arrive_expect_tx(
  cuda::ptx::sem_release_t sem,
  cuda::ptx::scope_t<Scope> scope,
  cuda::ptx::space_shared_cluster_t space,
  uint64_t* addr,
  const uint32_t& tx_count);
*/
#if __cccl_ptx_isa >= 800
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline void mbarrier_arrive_expect_tx(
  sem_release_t __sem,
  scope_t<_Scope> __scope,
  space_shared_cluster_t __space,
  _CUDA_VSTD::uint64_t* __addr,
  const _CUDA_VSTD::uint32_t& __tx_count)
{
  // __sem == sem_release (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  // __space == space_shared_cluster (due to parameter type constraint)



  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,(
    if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1; // 7. "
        :
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1; // 7. "
        :
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    }

  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    __device__ void __void__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
    return __void__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
  ));
}
#endif // __cccl_ptx_isa >= 800


// 9.7.12.15.14. Parallel Synchronization and Communication Instructions: mbarrier.arrive_drop
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop

// 9.7.12.15.15. Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive

// 9.7.12.15.16. Parallel Synchronization and Communication Instructions: mbarrier.test_wait/mbarrier.try_wait
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait

// 9.7.12.15.17. Parallel Synchronization and Communication Instructions: mbarrier.pending_count
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count

_LIBCUDACXX_END_NAMESPACE_CUDA_PTX

#endif // _CUDA_PTX_PARALLEL_SYNCHRONIZATION_AND_COMMUNICATION_INSTRUCTIONS_MBARRIER_H_
