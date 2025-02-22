//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__FUNCTIONAL_FOR_EACH_CANCELED_H
#define _CUDA__FUNCTIONAL_FOR_EACH_CANCELED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/cstdint>

#include <cooperative_groups.h>
#include <nv/target>

#if _CCCL_HAS_CUDA_COMPILER

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <int __ThreadBlockRank>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI dim3 __thread_block_indices() noexcept {
  dim3 __block_idx = dim3(blockIdx.x, 1, 1);
  if constexpr (__ThreadBlockRank >= 2)
  {
    __block_idx = dim3(blockIdx.x, blockIdx.y, 1);
  }
  if constexpr (__ThreadBlockRank >= 3)
  {
    __block_idx = dim3(blockIdx.x, blockIdx.y, blockIdx.z);
  }
  return __block_idx;
}

#  if _CCCL_HAS_INT128()

#    if __cccl_ptx_isa >= 870

template <int _Index>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI int __cluster_get_dim(__int128 __result) noexcept
{
  int __r;
  if constexpr (_Index == 0)
  {
    asm volatile("clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, %1;"
                 : "=r"(__r)
                 : "q"(__result)
                 : "memory");
  }
  else if constexpr (_Index == 1)
  {
    asm volatile("clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, %1;"
                 : "=r"(__r)
                 : "q"(__result)
                 : "memory");
  }
  else if constexpr (_Index == 2)
  {
    asm volatile("clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, %1;"
                 : "=r"(__r)
                 : "q"(__result)
                 : "memory");
  }
  else
  {
    _CCCL_UNREACHABLE();
  }
  return __r;
}

template <int __ThreadBlockRank>
_CCCL_NODISCARD _CCCL_DEVICE _CCCL_HIDE_FROM_ABI dim3 __try_cancel_result_to_dim3(__int128 __result) noexcept {
  dim3 __b(::cuda::__cluster_get_dim<0>(__result), 1, 1);
  if constexpr (__ThreadBlockRank >= 2)
    {
      __b.y = ::cuda::__cluster_get_dim<1>(__result);
    }
  if constexpr (__ThreadBlockRank == 3)
    {
      __b.z = ::cuda::__cluster_get_dim<2>(__result);
    }
  return __b;
}

template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void
__for_each_canceled_block_sm100(dim3 __block_idx, bool __is_leader, __UnaryFunction __uf)
{
  __shared__ uint64_t __barrier; // TODO: use 2 barriers and 2 results to avoid last sync threads
  __shared__ __int128 __result;
  bool __phase = false;

  // Initialize barrier and kick-start try_cancel pipeline:
  if (__is_leader)
  {
    auto __leader_mask = __activemask();
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      // elect.sync is a workaround for peeling loop (#nvbug-id)
      "elect.sync _|p, %2;\n\t"
      "@p mbarrier.init.shared::cta.b64 [%1], 1;\n\t"
      // `try_cancel` access the mbarrier using generic-proxy, so no cross-proxy fence required here
      "@p clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n\t"
      // This arrive does not order prior memory operations and can be relaxed.
      "@p mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%1], 16;\n\t"
      "}"
      :
      : "r"((int) __cvta_generic_to_shared(&__result)),
        "r"((int) __cvta_generic_to_shared(&__barrier)),
        "r"(__leader_mask)
      : "memory");
  }

  do
  {
    _CUDA_VSTD::invoke(__uf, __block_idx);
    if (__is_leader)
    {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "waitLoop:\n\t\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared.b64 p, [%0], %1;\n\t\t"
        "@!p bra waitLoop;\n\t"
	"@p mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], 16;\n\t"
        "}"
        :
        : "r"((int) __cvta_generic_to_shared(&__barrier)), "r"((unsigned) __phase)
        : "memory");
      __phase = !__phase;
    }
    __syncthreads(); // All threads of prior thread block have "exited".
    // Note: this syncthreads provides the .acquire.cta fence preventing
    // the next query operations from being re-ordered above the poll loop.
    {
      int __success = 0;
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, %1;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__success)
        : "q"(__result));
      if (__success != 1)
      {
        // Invalidating mbarrier and synchronizing before exiting not
        // required since each thread block calls this API at most once.
        break;
      }
    }

    __block_idx =  __try_cancel_result_to_dim3<__ThreadBlockRank>(__result);

    // Wait for all threads to read __result before issuing next async op.
    // generic->generic synchronization
    __syncthreads();
    // TODO: only control-warp requires sync, other warps can arrive
    // TODO: double-buffering results+barrier pairs using phase to avoids this sync

    if (__is_leader)
    {
      auto __leader_mask = __activemask();
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        // elect.sync is a workaround for peeling loop (#nvbug-id)
        "elect.sync _|p, %2;\n\t"
        // generic->async release + acquire synchronization of prior reads:
        // use bi-directional cross-proxy acq_rel fence instead of uni-dir rel; acq; fences.
        "@p fence.proxy.async.shared::cta;\n\t"
        // try to cancel another thread block
        "@p clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n\t"
        "}"
        :
        : "r"((int) __cvta_generic_to_shared(&__result)),
          "r"((int) __cvta_generic_to_shared(&__barrier)),
          "r"(__leader_mask)
        : "memory");
    }
  } while (true);
}

template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI  void
__for_each_canceled_cluster_sm100(dim3 __block_idx, bool __is_cluster_leader_block, bool __is_block_leader_thread, __UnaryFunction __uf) {
  __shared__ uint64_t __barrier; // TODO: use 2 barriers and 2 results avoid last sync threads
  __shared__ __int128 __result;
  bool __phase = false;
  dim3 __pos_in_cluster = cooperative_groups::cluster_group::block_index(); // TODO: optimize

  // Initialize barriers:
  if (__is_block_leader_thread)
  {
    auto __leader_mask = __activemask();
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      // elect.sync is a workaround for peeling loop (#nvbug-id)
      "elect.sync _|p, %2;\n\t"
      "@p mbarrier.init.shared::cta.b64 [%1], 1;\n\t"
      // Release mbarrier init to cluster scope.
      "@p fence.mbarrier_init.release.cluster;\n\t"      
      // This arrive does not order prior memory operations and can be relaxed.
      "@p mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%1], 16;\n\t"
      "}"
      :
      : "r"((int) __cvta_generic_to_shared(&__result)),
        "r"((int) __cvta_generic_to_shared(&__barrier)),
        "r"(__leader_mask)
      : "memory");
  }

  // Synchronize barrier initialization across cluster:
  asm volatile(
      "{\n\t"
        "barrier.cluster.arrive.relaxed;\n\t"
        "barrier.cluster.wait.relaxed;\n\t"
      "}"
  );

  // Kickstart pipeline:
  if (__is_cluster_leader_block && __is_block_leader_thread)
  {
    auto __leader_mask = __activemask();
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      // elect.sync is a workaround for peeling loop (#nvbug-id)
      "elect.sync _|p, %2;\n\t"
      // Note: only cluster block leader thread needs to acquire mbarrier initialization.
      // Leader threads of other blocks don't have to because they only access their local mbarriers.
      "@p fence.acquire.shared::cta.cluster;\n\t"
      // At this point, the initialization of mbarriers by all blocks is visible to the lead cluster block.
      // `try_cancel` access all mbarriers in cluster using generic-proxy, so no cross-proxy fence required here
      "@p clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
      "}"
      :
      : "r"((int) __cvta_generic_to_shared(&__result)),
        "r"((int) __cvta_generic_to_shared(&__barrier)),
        "r"(__leader_mask)
      : "memory");
  }
  
  do {
    _CUDA_VSTD::invoke(__uf, __block_idx);

    if (__is_block_leader_thread)
    {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "waitLoop:\n\t\t"
        "mbarrier.try_wait.parity.relaxed.cluster.shared.b64 p, [%0], %1;\n\t\t"
        "@!p bra waitLoop;\n\t"
	// Issue the expect_tx for the next round:
	"@p mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%1], 16;\n\t"
	// Async->Generic cluster-scope acquire for the local shared memory try_cancel result:
	"@p fence.acquire.sync_restrict::shared::cta.cluster;\n\t"
        "}"
        :
        : "r"((int) __cvta_generic_to_shared(&__barrier)), "r"((unsigned) __phase)
        : "memory");
      __phase = !__phase;
    }
    __syncthreads(); // All threads of prior thread block have "exited".
    // Note: this syncthreads provides the .acquire.cta fence preventing
    // the next query operations from being re-ordered above the poll loop.
      
    {
      int __success = 0;
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, %1;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__success)
        : "q"(__result));
      if (__success != 1)
      {
        // Invalidating mbarrier and synchronizing before exiting not
        // required since each thread block calls this API at most once.
        break;
      }
    }
    __block_idx = __try_cancel_result_to_dim3<__ThreadBlockRank>(__result);
    // The block_idx loaded is from the 0-th block in the cluster:
    __block_idx.x += __pos_in_cluster.x;
    if constexpr (__ThreadBlockRank >= 2) {
      __block_idx.y += __pos_in_cluster.y;
    }
    if constexpr (__ThreadBlockRank == 3) {
      __block_idx.z += __pos_in_cluster.y;
    }

    // TODO: double-buffering results+barrier pairs using phase avoids this sync

    // Wait for all threads in the cluster to read __result before issuing next async op.
    // generic->async release of result reads:
    asm volatile(
      "{\n\t"
      "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;\n\t"
      "barrier.cluster.arrive.relaxed;\n\t"
      "barrier.cluster.wait.relaxed;\n\t" // TODO: move to below
      "}"
    );

    // Only the leader thread from leader block needs to wait on the barrier
    if (__is_cluster_leader_block && __is_block_leader_thread) {
      auto __leader_mask = __activemask();
      asm volatile(
	"{\n\t"
	".reg .pred p;\n\t"
	// elect.sync is a workaround for peeling loop (#nvbug-id)
	"elect.sync _|p, %2;\n\t"
	// Wait at the cluster barrier
	// "@p barrier.cluster.wait.relaxed;\n\t"
	// generic->async acquire of result reads
	"@p fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;"
	// try to cancel another thread block
	"@p clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];\n\t"
	"}"
	:
	: "r"((int)__cvta_generic_to_shared(&__result)), "r"((int)__cvta_generic_to_shared(&__barrier)), "r"(__leader_mask)
	: "memory"
      );
    }
  } while (true);
}

#    else // ^^^ __cccl_ptx_isa >= 870 ^^^ / vvv __cccl_ptx_isa < 870 vvv

template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void
__for_each_canceled_block_sm100(dim3 __block_idx, bool __is_leader, __UnaryFunction __uf)
{
  // We are compiling for SM100 but PTX 8.7 is not supported, so fall back to just calling the function
  _CUDA_VSTD::invoke(_CUDA_VSTD::move(__uf), __block_idx);
}

template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void
__for_each_canceled_cluster_sm100(dim3 __block_idx, bool __is_cluster_leader_block, bool __is_block_leader_thread, __UnaryFunction __uf)
{
  // We are compiling for SM100 but PTX 8.7 is not supported, so fall back to just calling the function
  _CUDA_VSTD::invoke(_CUDA_VSTD::move(__uf), __block_idx);
}

#    endif // __cccl_ptx_isa <= 870
#  endif // _CCCL_HAS_INT128()

//! Internal version of `for_each_canceled_block` that enables caller to select control thread
//! by passing `__is_leader` equals `true` on that thread.
//!
//! Preconditions over for_each_canceled_work:
//! - Exactly one thread in the thread block calls this API with `__is_leader` equals `true`;
//!   all other threads in the block shall call this API with `__is_leader` equals ` false.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void __for_each_canceled_block(bool __is_leader, __UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3,
                "__for_each_canceled_block<ThreadBlockRank>: ThreadBlockRank out-of-range [1, 3].");
  static_assert(_CUDA_VSTD::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "__for_each_canceled_block first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: __for_each_canceled_block([](dim3 block_idx) { ... });");
  dim3 __block_idx = __thread_block_indices<__ThreadBlockRank>();
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_100,
                     (::cuda::__for_each_canceled_block_sm100(__block_idx, __is_leader, _CUDA_VSTD::move(__uf));),
                     NV_ANY_TARGET,
                     (_CUDA_VSTD::invoke(_CUDA_VSTD::move(__uf), __block_idx);))
}

//! See extended_api/work_stealing.rst for documentation on this public API.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void for_each_canceled_block(__UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3,
                "for_each_canceled_block<ThreadBlockRank>: ThreadBlockRank out-of-range [1, 3].");
  static_assert(_CUDA_VSTD::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "for_each_canceled_block first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: for_each_canceled_block([](dim3 block_idx) { ... });");
  if constexpr (__ThreadBlockRank == 1)
  {
    bool __leader_thread = threadIdx.x == 0;
    ::cuda::__for_each_canceled_block<1>(__leader_thread, _CUDA_VSTD::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 2)
  {
    bool __leader_thread = threadIdx.x == 0 && threadIdx.y == 0;
    ::cuda::__for_each_canceled_block<2>(__leader_thread, _CUDA_VSTD::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 3)
  {
    bool __leader_thread = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
    ::cuda::__for_each_canceled_block<3>(__leader_thread, _CUDA_VSTD::move(__uf));
  }
}

//! Internal version of `for_each_canceled_cluster` that enables caller to select control thread
//! by passing `__is_leader_thread` on that thread.
//!
//! Preconditions over for_each_canceled_cluster:
//! - Exactly one thread in the thread block cluster calls this API with `__is_leader` equals `true`;
//!   all other threads in the cluster shall call this API with `__is_leader` equals ` false.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void __for_each_canceled_cluster(bool __is_cluster_leader_block, bool __is_block_leader_thread, __UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3,
                "__for_each_canceled_cluster<ThreadBlockRank>: ThreadBlockRank out-of-range [1, 3].");
  static_assert(_CUDA_VSTD::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "__for_each_canceled_cluster first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: __for_each_canceled_block([](dim3 block_idx) { ... });");
  dim3 __block_idx = __thread_block_indices<__ThreadBlockRank>();
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_100,
                     (::cuda::__for_each_canceled_cluster_sm100(__block_idx, __is_cluster_leader_block, __is_block_leader_thread, _CUDA_VSTD::move(__uf));),
                     NV_ANY_TARGET,
                     (_CUDA_VSTD::invoke(_CUDA_VSTD::move(__uf), __block_idx);))
}

//! See extended_api/work_stealing.rst for documentation on this public API.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void for_each_canceled_cluster(__UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3,
                "for_each_canceled_cluster<ThreadBlockRank>: ThreadBlockRank out-of-range [1, 3].");
  static_assert(_CUDA_VSTD::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "for_each_canceled_block first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: for_each_canceled_block([](dim3 block_idx) { ... });");
  dim3 __pos_in_cluster = cooperative_groups::cluster_group::block_index(); // TODO: optimize
  if constexpr (__ThreadBlockRank == 1)
  {
    bool __leader_block = __pos_in_cluster.x == 0;
    bool __leader_thread = threadIdx.x == 0;
    ::cuda::__for_each_canceled_cluster<1>(__leader_block, __leader_thread, _CUDA_VSTD::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 2)
  {
    bool __leader_block = __pos_in_cluster.x == 0 && __pos_in_cluster.y == 0;
    bool __leader_thread = threadIdx.x == 0 && threadIdx.y == 0;
    ::cuda::__for_each_canceled_cluster<2>(__leader_block, __leader_thread, _CUDA_VSTD::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 3)
  {
    bool __leader_block = __pos_in_cluster.x == 0 && __pos_in_cluster.y == 0 && __pos_in_cluster.z == 0;
    bool __leader_thread = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
    ::cuda::__for_each_canceled_cluster<3>(__leader_block, __leader_thread, _CUDA_VSTD::move(__uf));
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_HAS_CUDA_COMPILER

#endif // _CUDA__FUNCTIONAL_FOR_EACH_CANCELED_H
