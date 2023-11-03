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

#if defined(_CCCL_COMPILER_NVHPC) && defined(_CCCL_USE_IMPLICIT_SYSTEM_DEADER)
#pragma GCC system_header
#else // ^^^ _CCCL_COMPILER_NVHPC ^^^ / vvv !_CCCL_COMPILER_NVHPC vvv
_CCCL_IMPLICIT_SYSTEM_HEADER
#endif // !_CCCL_COMPILER_NVHPC

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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
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
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_80__();
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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
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
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
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
        "mbarrier.arrive.release.cta.shared.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.release.cluster.shared.b64                   %0,  [%1];           // 3a. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr))
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive(
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
        "mbarrier.arrive.release.cta.shared.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.release.cluster.shared.b64                   %0,  [%1], %2;    // 3b. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__count)
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void mbarrier_arrive(
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
      : "r"(__as_ptr_smem(__addr))
      : "memory"
    );

  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    return __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void mbarrier_arrive(
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
      : "r"(__as_ptr_smem(__addr)),
        "r"(__count)
      : "memory"
    );

  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    return __void__cuda_ptx_mbarrier_arrive_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_no_complete(
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
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_no_complete_is_not_supported_before_SM_80__();
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
extern "C" _LIBCUDACXX_DEVICE _CUDA_VSTD::uint64_t ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_LIBCUDACXX_DEVICE static inline _CUDA_VSTD::uint64_t mbarrier_arrive_expect_tx(
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
        "mbarrier.arrive.expect_tx.release.cta.shared.b64 %0, [%1], %2; // 8. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    } else if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cluster) {
      asm (
        "mbarrier.arrive.expect_tx.release.cluster.shared.b64 %0, [%1], %2; // 8. "
        : "=l"(__state)
        : "r"(__as_ptr_smem(__addr)),
          "r"(__tx_count)
        : "memory"
      );
    }
    return __state;
  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
    return ___cuda_vstd_uint64_t__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
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
extern "C" _LIBCUDACXX_DEVICE void __void__cuda_ptx_mbarrier_arrive_expect_tx_is_not_supported_before_SM_90__();
template <typename=void>
_LIBCUDACXX_DEVICE static inline void mbarrier_arrive_expect_tx(
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
      : "r"(__as_ptr_smem(__addr)),
        "r"(__tx_count)
      : "memory"
    );

  ),(
    // Unsupported architectures will have a linker error with a semi-decent error message
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
