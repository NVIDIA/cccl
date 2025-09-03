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

#include <nv/target>

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_HAS_INT128() && __cccl_ptx_isa >= 870

template <int _Index>
[[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI int __cluster_get_dim(__int128 __result) noexcept
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

//! This API for implementing work-stealing, repeatedly attempts to cancel the launch of a thread block
//! from the current grid. On success, it invokes the unary function `__uf` before trying again.
//! On failure, it returns.
//!
//! This API does not provide any memory synchronization.
//! This API does not guarantee that any thread will invoke `__uf` with the next block index until all
//! invocatons of `__uf` for the prior block index have returned.
//!
//! Preconditions:
//! - All thread block threads shall call this API exactly once.
//! - Exactly one thread block thread shall call this API with `__is_leader` equals `true`.
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
    auto __leader_mask = ::__activemask();
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
      : "r"((int) ::__cvta_generic_to_shared(&__result)),
        "r"((int) ::__cvta_generic_to_shared(&__barrier)),
        "r"(__leader_mask)
      : "memory");
  }

  do
  {
    ::cuda::std::invoke(__uf, __block_idx);
    if (__is_leader)
    {
      asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "waitLoop:\n\t\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared.b64 p, [%0], %1;\n\t\t"
        "@!p bra waitLoop;\n\t"
        "}"
        :
        : "r"((int) ::__cvta_generic_to_shared(&__barrier)), "r"((unsigned) __phase)
        : "memory");
      __phase = !__phase;
    }
    ::__syncthreads(); // All threads of prior thread block have "exited".
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

    // Read new thread block dimensions
    dim3 __b(::cuda::__cluster_get_dim<0>(__result), 1, 1);
    if constexpr (__ThreadBlockRank >= 2)
    {
      __b.y = ::cuda::__cluster_get_dim<1>(__result);
    }
    if constexpr (__ThreadBlockRank == 3)
    {
      __b.z = ::cuda::__cluster_get_dim<2>(__result);
    }
    __block_idx = __b;

    // Wait for all threads to read __result before issuing next async op.
    // generic->generic synchronization
    ::__syncthreads();
    // TODO: only control-warp requires sync, other warps can arrive
    // TODO: double-buffering results+barrier pairs using phase to avoids this sync

    if (__is_leader)
    {
      auto __leader_mask = ::__activemask();
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
        "@p mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%1], 16;\n\t"
        "}"
        :
        : "r"((int) ::__cvta_generic_to_shared(&__result)),
          "r"((int) ::__cvta_generic_to_shared(&__barrier)),
          "r"(__leader_mask)
        : "memory");
    }
  } while (true);
}

#  else // ^^^ _CCCL_HAS_INT128() && __cccl_ptx_isa >= 870 ^^^ / vvv !_CCCL_HAS_INT128() || __cccl_ptx_isa < 870 vvv
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void
__for_each_canceled_block_sm100(dim3 __block_idx, bool __is_leader, __UnaryFunction __uf)
{
  // We are compiling for SM100 but PTX 8.7 is not supported, so fall back to just calling the function
  ::cuda::std::invoke(::cuda::std::move(__uf), __block_idx);
}
#  endif // _CCCL_HAS_INT128() && __cccl_ptx_isa >= 870

//! This API for implementing work-stealing, repeatedly attempts to cancel the launch of a thread block
//! from the current grid. On success, it invokes the unary function `__uf` before trying again.
//! On failure, it returns.
//!
//! This API does not provide any memory synchronization.
//! This API does not guarantee that any thread will invoke `__uf` with the next block index until all
//! invocatons of `__uf` for the prior block index have returned.
//!
//! Preconditions:
//! - All thread block threads shall call this API exactly once.
//! - Exactly one thread block thread shall call this API with `__is_leader` equals `true`.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void __for_each_canceled_block(bool __is_leader, __UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3, "ThreadBlockRank out-of-range [1, 3].");
  static_assert(::cuda::std::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "__for_each_canceled_block first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: __for_each_canceled_block([](dim3 block_idx) { ... });");
  dim3 __block_idx = dim3(blockIdx.x, 1, 1);
  if constexpr (__ThreadBlockRank >= 2)
  {
    __block_idx = dim3(blockIdx.x, blockIdx.y, 1);
  }
  if constexpr (__ThreadBlockRank >= 3)
  {
    __block_idx = dim3(blockIdx.x, blockIdx.y, blockIdx.z);
  }

  NV_DISPATCH_TARGET(NV_PROVIDES_SM_100,
                     (::cuda::__for_each_canceled_block_sm100(__block_idx, __is_leader, ::cuda::std::move(__uf));),
                     NV_ANY_TARGET,
                     (::cuda::std::invoke(::cuda::std::move(__uf), __block_idx);))
}

//! This API used to implement work-stealing, repeatedly attempts to cancel the launch of a thread block
//! from the current grid. On success, it invokes the unary function `__uf` before trying again.
//! On failure, it returns.
//!
//! This API does not provide any memory synchronization.
//! This API does not guarantee that any thread will invoke `__uf` with the next block index until all
//! invocatons of `__uf` for the prior block index have returned.
//!
//! Preconditions:
//! - All thread block threads shall call this API exactly once.
//! - Exactly one thread block thread shall call this API with `__is_leader` equals `true`.
template <int __ThreadBlockRank = 3, typename __UnaryFunction = void>
_CCCL_DEVICE _CCCL_HIDE_FROM_ABI void for_each_canceled_block(__UnaryFunction __uf)
{
  static_assert(__ThreadBlockRank >= 1 && __ThreadBlockRank <= 3,
                "for_each_canceled_block<ThreadBlockRank>: ThreadBlockRank out-of-range [1, 3].");
  static_assert(::cuda::std::is_invocable_r_v<void, __UnaryFunction, dim3>,
                "for_each_canceled_block first argument requires an UnaryFunction with signature: void(dim3).\n"
                "For example, call with lambda: for_each_canceled_block([](dim3 block_idx) { ... });");
  if constexpr (__ThreadBlockRank == 1)
  {
    ::cuda::__for_each_canceled_block<1>(threadIdx.x == 0, ::cuda::std::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 2)
  {
    ::cuda::__for_each_canceled_block<2>(threadIdx.x == 0 && threadIdx.y == 0, ::cuda::std::move(__uf));
  }
  else if constexpr (__ThreadBlockRank == 3)
  {
    ::cuda::__for_each_canceled_block<3>(
      threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0, ::cuda::std::move(__uf));
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA__FUNCTIONAL_FOR_EACH_CANCELED_H
