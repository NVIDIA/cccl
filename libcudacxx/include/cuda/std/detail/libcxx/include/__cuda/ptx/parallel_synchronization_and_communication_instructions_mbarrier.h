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

#if __cccl_ptx_sm >= 900 && __cccl_ptx_isa >= 780
template <dot_scope _Sco>
_LIBCUDACXX_DEVICE inline _CUDA_VSTD::uint64_t mbarrier_arrive_expect_tx(
  sem_release_t __sem,
  scope_t<_Sco> __scope,
  space_shared_t __spc,
  _CUDA_VSTD::uint64_t* __addr,
  _CUDA_VSTD::uint32_t __tx_count)
{
  // Arrive on local shared memory barrier
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  _CUDA_VSTD::uint64_t __token;

  if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(__token)
        : "r"(__as_smem_ptr(__addr)),
          "r"(__tx_count)
        : "memory");
    } else {
    asm (
      "mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 %0, [%1], %2;"
      : "=l"(__token)
      : "r"(__as_smem_ptr(__addr)),
        "r"(__tx_count)
      : "memory");
  }
  return __token;
}
#endif // __cccl_ptx_isa

#if __cccl_ptx_sm >= 900 && __cccl_ptx_isa >= 780
template <dot_scope _Sco>
_LIBCUDACXX_DEVICE inline void mbarrier_arrive_expect_tx(
  sem_release_t __sem,
  scope_t<_Sco> __scope,
  space_shared_cluster_t __spc,
  _CUDA_VSTD::uint64_t* __addr,
  _CUDA_VSTD::uint32_t __tx_count)
{
  // Arrive on remote cluster barrier
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  if _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 (__scope == scope_cta) {
      asm (
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
        :
        : "r"(__as_smem_ptr(__addr)),
          "r"(__tx_count)
        : "memory");
    } else {
    asm (
      "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;"
      :
      : "r"(__as_smem_ptr(__addr)),
        "r"(__tx_count)
      : "memory");
  }
}
#endif // __cccl_ptx_isa


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
