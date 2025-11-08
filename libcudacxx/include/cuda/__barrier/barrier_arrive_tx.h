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

#ifndef _CUDA_PTX_BARRIER_ARRIVE_TX_H_
#define _CUDA_PTX_BARRIER_ARRIVE_TX_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  if __cccl_ptx_isa >= 800

#    include <cuda/__barrier/barrier_block_scope.h>
#    include <cuda/__memory/address_space.h>
#    include <cuda/__ptx/instructions/mbarrier_arrive.h>
#    include <cuda/__ptx/ptx_dot_variants.h>
#    include <cuda/__ptx/ptx_helper_functions.h>
#    include <cuda/std/__atomic/scopes.h>
#    include <cuda/std/cstdint>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

extern "C" _CCCL_DEVICE void __cuda_ptx_barrier_arrive_tx_is_not_supported_before_SM_90__();
[[nodiscard]] _CCCL_DEVICE inline barrier<thread_scope_block>::arrival_token barrier_arrive_tx(
  barrier<thread_scope_block>& __b,
  ::cuda::std::ptrdiff_t __arrive_count_update,
  ::cuda::std::ptrdiff_t __transaction_count_update)
{
  _CCCL_ASSERT(
    ::cuda::device::is_address_from(::cuda::device::barrier_native_handle(__b), ::cuda::device::address_space::shared),
    "Barrier must be located in local shared memory.");
  _CCCL_ASSERT(1 <= __arrive_count_update, "Arrival count update must be at least one.");
  _CCCL_ASSERT(__arrive_count_update <= (1 << 20) - 1, "Arrival count update cannot exceed 2^20 - 1.");
  _CCCL_ASSERT(__transaction_count_update >= 0, "Transaction count update must be non-negative.");
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#contents-of-the-mbarrier-object
  _CCCL_ASSERT(__transaction_count_update <= (1 << 20) - 1, "Transaction count update cannot exceed 2^20 - 1.");

  barrier<thread_scope_block>::arrival_token __token = {};
  // On architectures pre-sm90, arrive_tx is not supported.
  // We do not check for the statespace of the barrier here. This is
  // on purpose. This allows debugging tools like memcheck/racecheck
  // to detect that we are passing a pointer with the wrong state
  // space to mbarrier.arrive. If we checked for the state space here,
  // and __trap() if wrong, then those tools would not be able to help
  // us in release builds. In debug builds, the error would be caught
  // by the asserts at the top of this function.
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (

      auto __native_handle = ::cuda::device::barrier_native_handle(__b);
      auto __bh            = ::__cvta_generic_to_shared(__native_handle);
      if (__arrive_count_update == 1) {
        __token = ::cuda::ptx::mbarrier_arrive_expect_tx(
          ::cuda::ptx::sem_release,
          ::cuda::ptx::scope_cta,
          ::cuda::ptx::space_shared,
          __native_handle,
          __transaction_count_update);
      } else {
        asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
            :
            : "r"(static_cast<::cuda::std::uint32_t>(__bh)),
              "r"(static_cast<::cuda::std::uint32_t>(__transaction_count_update))
            : "memory");
        __token = ::cuda::ptx::mbarrier_arrive(
          ::cuda::ptx::sem_release,
          ::cuda::ptx::scope_cta,
          ::cuda::ptx::space_shared,
          __native_handle,
          __arrive_count_update);
      }),
    (::cuda::device::__cuda_ptx_barrier_arrive_tx_is_not_supported_before_SM_90__();));
  return __token;
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA_PTX_BARRIER_ARRIVE_TX_H_
