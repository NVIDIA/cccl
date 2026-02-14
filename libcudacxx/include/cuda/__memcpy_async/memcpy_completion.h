//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_MEMCPY_COMPLETION_H
#define _CUDA___MEMCPY_ASYNC_MEMCPY_COMPLETION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/async_contract_fulfillment.h>
#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/__barrier/barrier_expect_tx.h>
#include <cuda/__fwd/pipeline.h>
#include <cuda/__memcpy_async/completion_mechanism.h>
#include <cuda/__memcpy_async/is_local_smem_barrier.h>
#include <cuda/__memcpy_async/try_get_barrier_handle.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/cstdint>

#if _CCCL_CUDA_COMPILATION()
#  include <cuda/__ptx/ptx_dot_variants.h>
#  include <cuda/__ptx/ptx_helper_functions.h>
#endif // _CCCL_CUDA_COMPILATION()

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// This struct contains functions to defer the completion of a barrier phase
// or pipeline stage until a specific memcpy_async operation *initiated by
// this thread* has completed.

// The user is still responsible for arriving and waiting on (or otherwise
// synchronizing with) the barrier or pipeline barrier to see the results of
// copies from other threads participating in the synchronization object.
struct __memcpy_completion_impl
{
  template <typename _Group>
  [[nodiscard]] _CCCL_API inline static async_contract_fulfillment
  __defer(__completion_mechanism __cm,
          _Group const& __group,
          ::cuda::std::size_t __size,
          barrier<::cuda::thread_scope_block>& __barrier)
  {
    // In principle, this is the overload for shared memory barriers. However, a
    // block-scope barrier may also be located in global memory. Therefore, we
    // check if the barrier is a non-smem barrier and handle that separately.
    if (!::cuda::__is_local_smem_barrier(__barrier))
    {
      return __defer_non_smem_barrier(__cm, __group, __size, __barrier);
    }

    switch (__cm)
    {
      case __completion_mechanism::__async_group:
        // Pre-SM80, the async_group mechanism is not available.
        NV_IF_TARGET(
          NV_PROVIDES_SM_80,
          (
            // Non-Blocking: unbalance barrier by 1, barrier will be
            // rebalanced when all thread-local cp.async instructions
            // have completed writing to shared memory.
            ::cuda::std::uint64_t* __bh = ::cuda::__try_get_barrier_handle(__barrier);

            asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" ::"r"(
              static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(__bh))) : "memory");));
        return async_contract_fulfillment::async;
      case __completion_mechanism::__async_bulk_group:
        // This completion mechanism should not be used with a shared
        // memory barrier. Or at least, we do not currently envision
        // bulk group to be used with shared memory barriers.
        _CCCL_UNREACHABLE();
      case __completion_mechanism::__mbarrier_complete_tx:
        // we already updated the mbarrier's tx count when we issued the bulk copy
        return async_contract_fulfillment::async;
      case __completion_mechanism::__sync:
        // sync: In this case, we do not need to do anything. The user will have
        // to issue `bar.arrive_wait();` to see the effect of the transaction.
        return async_contract_fulfillment::none;
      default:
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
    }
  }

  template <typename _Group, thread_scope _Sco, typename _CompF>
  [[nodiscard]] _CCCL_API inline static async_contract_fulfillment __defer(
    __completion_mechanism __cm, _Group const& __group, ::cuda::std::size_t __size, barrier<_Sco, _CompF>& __barrier)
  {
    return __defer_non_smem_barrier(__cm, __group, __size, __barrier);
  }

  template <typename _Group, thread_scope _Sco, typename _CompF>
  [[nodiscard]] _CCCL_API inline static async_contract_fulfillment
  __defer_non_smem_barrier(__completion_mechanism __cm, _Group const&, ::cuda::std::size_t, barrier<_Sco, _CompF>&)
  {
    // Overload for non-smem barriers.
    switch (__cm)
    {
      case __completion_mechanism::__async_group:
        // Pre-SM80, the async_group mechanism is not available.
        NV_IF_TARGET(NV_PROVIDES_SM_80,
                     (
                       // Blocking: wait for all thread-local cp.async instructions to have
                       // completed writing to shared memory.
                       asm volatile("cp.async.wait_all;" :: : "memory");));
        return async_contract_fulfillment::async;
      case __completion_mechanism::__mbarrier_complete_tx:
        // Non-smem barriers do not have an mbarrier_complete_tx mechanism..
        _CCCL_UNREACHABLE();
      case __completion_mechanism::__async_bulk_group:
        // This completion mechanism is currently not expected to be used with barriers.
        _CCCL_UNREACHABLE();
      case __completion_mechanism::__sync:
        // sync: In this case, we do not need to do anything.
        return async_contract_fulfillment::none;
      default:
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
    }
  }

  template <typename _Group, thread_scope _Sco>
  [[nodiscard]] _CCCL_API inline static async_contract_fulfillment
  __defer(__completion_mechanism __cm, _Group const&, ::cuda::std::size_t, pipeline<_Sco>&)
  {
    switch (__cm)
    {
      case __completion_mechanism::__async_group:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__async_bulk_group:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__mbarrier_complete_tx:
        return async_contract_fulfillment::async;
      case __completion_mechanism::__sync:
        return async_contract_fulfillment::none;
      default:
        // Get rid of "control reaches end of non-void function":
        _CCCL_UNREACHABLE();
    }
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_MEMCPY_COMPLETION_H
