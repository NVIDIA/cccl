// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_BARRIER_H
#define _LIBCUDACXX___CUDA_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 700
#  error "CUDA synchronization primitives are only supported for sm_70 and up."
#endif

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/aligned_size.h>
#include <cuda/__barrier/async_contract_fulfillment.h>
#include <cuda/__barrier/barrier.h>
#include <cuda/__barrier/barrier_arrive_tx.h>
#include <cuda/__barrier/barrier_expect_tx.h>
#include <cuda/__barrier/barrier_native_handle.h>
#include <cuda/__barrier/barrier_thread_scope_block.h>
#include <cuda/__barrier/completion_mechanism.h>
#include <cuda/__fwd/pipeline.h>
#include <cuda/std/__atomic/api/owned.h>
#include <cuda/std/__type_traits/void_t.h> // _CUDA_VSTD::void_t

#if defined(_CCCL_CUDA_COMPILER)
#  include <cuda/ptx> // cuda::ptx::*
#endif // _CCCL_CUDA_COMPILER

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

#if defined(_CCCL_CUDA_COMPILER)

#  if __cccl_ptx_isa >= 800

extern "C" _CCCL_DEVICE void __cuda_ptx_memcpy_async_tx_is_not_supported_before_SM_90__();
template <typename _Tp, _CUDA_VSTD::size_t _Alignment>
_CCCL_DEVICE inline async_contract_fulfillment memcpy_async_tx(
  _Tp* __dest,
  const _Tp* __src,
  ::cuda::aligned_size_t<_Alignment> __size,
  ::cuda::barrier<::cuda::thread_scope_block>& __b)
{
  // When compiling with NVCC and GCC 4.8, certain user defined types that _are_ trivially copyable are
  // incorrectly classified as not trivially copyable. Remove this assertion to allow for their usage with
  // memcpy_async when compiling with GCC 4.8.
  // FIXME: remove the #if once GCC 4.8 is no longer supported.
#    if !defined(_CCCL_COMPILER_GCC) || _GNUC_VER > 408
  static_assert(_CUDA_VSTD::is_trivially_copyable<_Tp>::value, "memcpy_async_tx requires a trivially copyable type");
#    endif
  static_assert(16 <= _Alignment, "mempcy_async_tx expects arguments to be at least 16 byte aligned.");

  _CCCL_ASSERT(__isShared(barrier_native_handle(__b)), "Barrier must be located in local shared memory.");
  _CCCL_ASSERT(__isShared(__dest), "dest must point to shared memory.");
  _CCCL_ASSERT(__isGlobal(__src), "src must point to global memory.");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      if (__isShared(__dest) && __isGlobal(__src)) {
        _CUDA_VPTX::cp_async_bulk(
          _CUDA_VPTX::space_cluster,
          _CUDA_VPTX::space_global,
          __dest,
          __src,
          static_cast<uint32_t>(__size),
          barrier_native_handle(__b));
      } else {
        // memcpy_async_tx only supports copying from global to shared
        // or from shared to remote cluster dsmem. To copy to remote
        // dsmem, we need to arrive on a cluster-scoped barrier, which
        // is not yet implemented. So we trap in this case as well.
        _CCCL_UNREACHABLE();
      }),
    (__cuda_ptx_memcpy_async_tx_is_not_supported_before_SM_90__();));

  return async_contract_fulfillment::async;
}
#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILER

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#if defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <>
class barrier<thread_scope_thread, _CUDA_VSTD::__empty_completion> : private barrier<thread_scope_block>
{
  using __base = barrier<thread_scope_block>;

public:
  using __base::__base;

  _LIBCUDACXX_HIDE_FROM_ABI friend void
  init(barrier* __b,
       _CUDA_VSTD::ptrdiff_t __expected,
       _CUDA_VSTD::__empty_completion __completion = _CUDA_VSTD::__empty_completion())
  {
    init(static_cast<__base*>(__b), __expected, __completion);
  }

  using __base::arrive;
  using __base::arrive_and_drop;
  using __base::arrive_and_wait;
  using __base::max;
  using __base::wait;
};

template <typename... _Ty>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __unused(_Ty...)
{
  return true;
}

template <typename _Ty>
_LIBCUDACXX_HIDE_FROM_ABI constexpr bool __unused(_Ty&)
{
  return true;
}

// __is_local_smem_barrier returns true if barrier is (1) block-scoped and (2) located in shared memory.
template <thread_scope _Sco,
          typename _CompF,
          bool _Is_mbarrier = (_Sco == thread_scope_block)
                           && _CUDA_VSTD::is_same<_CompF, _CUDA_VSTD::__empty_completion>::value>
_LIBCUDACXX_HIDE_FROM_ABI bool __is_local_smem_barrier(barrier<_Sco, _CompF>& __barrier)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return _Is_mbarrier && __isShared(&__barrier);), (return false;));
}

// __try_get_barrier_handle returns barrier handle of block-scoped barriers and a nullptr otherwise.
template <thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t* __try_get_barrier_handle(barrier<_Sco, _CompF>& __barrier)
{
  return nullptr;
}

template <>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t*
__try_get_barrier_handle<::cuda::thread_scope_block, _CUDA_VSTD::__empty_completion>(
  barrier<::cuda::thread_scope_block>& __barrier)
{
  (void) __barrier;
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (return ::cuda::device::barrier_native_handle(__barrier);), NV_ANY_TARGET, (return nullptr;));
}

// This struct contains functions to defer the completion of a barrier phase
// or pipeline stage until a specific memcpy_async operation *initiated by
// this thread* has completed.

// The user is still responsible for arriving and waiting on (or otherwise
// synchronizing with) the barrier or pipeline barrier to see the results of
// copies from other threads participating in the synchronization object.
struct __memcpy_completion_impl
{
  template <typename _Group>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment
  __defer(__completion_mechanism __cm,
          _Group const& __group,
          _CUDA_VSTD::size_t __size,
          barrier<::cuda::thread_scope_block>& __barrier)
  {
    // In principle, this is the overload for shared memory barriers. However, a
    // block-scope barrier may also be located in global memory. Therefore, we
    // check if the barrier is a non-smem barrier and handle that separately.
    if (!__is_local_smem_barrier(__barrier))
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
            _CUDA_VSTD::uint64_t* __bh = __try_get_barrier_handle(__barrier);

            asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];" ::"r"(static_cast<_CUDA_VSTD::uint32_t>(
              __cvta_generic_to_shared(__bh)))
                         : "memory");));
        return async_contract_fulfillment::async;
      case __completion_mechanism::__async_bulk_group:
        // This completion mechanism should not be used with a shared
        // memory barrier. Or at least, we do not currently envision
        // bulk group to be used with shared memory barriers.
        _CCCL_UNREACHABLE();
      case __completion_mechanism::__mbarrier_complete_tx:
#  if __cccl_ptx_isa >= 800
        // Pre-sm90, the mbarrier_complete_tx completion mechanism is not available.
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (
                       // Only perform the expect_tx operation with the leader thread
                       if (__group.thread_rank() == 0) { ::cuda::device::barrier_expect_tx(__barrier, __size); }));
#  endif // __cccl_ptx_isa >= 800
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
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment __defer(
    __completion_mechanism __cm, _Group const& __group, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF>& __barrier)
  {
    return __defer_non_smem_barrier(__cm, __group, __size, __barrier);
  }

  template <typename _Group, thread_scope _Sco, typename _CompF>
  _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment __defer_non_smem_barrier(
    __completion_mechanism __cm, _Group const& __group, _CUDA_VSTD::size_t __size, barrier<_Sco, _CompF>& __barrier)
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
                       asm volatile("cp.async.wait_all;" ::
                                      : "memory");));
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
  _LIBCUDACXX_HIDE_FROM_ABI static async_contract_fulfillment
  __defer(__completion_mechanism __cm, _Group const& __group, _CUDA_VSTD::size_t __size, pipeline<_Sco>& __pipeline)
  {
    // pipeline does not sync on memcpy_async, defeat pipeline purpose otherwise
    __unused(__pipeline);
    __unused(__size);
    __unused(__group);

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

/***********************************************************************
 * memcpy_async code:
 *
 * A call to cuda::memcpy_async(dest, src, size, barrier) can dispatch to any of
 * these PTX instructions:
 *
 * 1. normal synchronous copy (fallback)
 * 2. cp.async:      shared  <- global
 * 3. cp.async.bulk: shared  <- global
 * 4. TODO: cp.async.bulk: global  <- shared
 * 5. TODO: cp.async.bulk: cluster <- shared
 *
 * Which of these options is chosen, depends on:
 *
 * 1. The alignment of dest, src, and size;
 * 2. The direction of the copy
 * 3. The current compute capability
 * 4. The requested completion mechanism
 *
 * PTX has 3 asynchronous completion mechanisms:
 *
 * 1. Async group           - local to a thread. Used by cp.async
 * 2. Bulk async group      - local to a thread. Used by cp.async.bulk (shared -> global)
 * 3. mbarrier::complete_tx - shared memory barier. Used by cp.async.bulk (other directions)
 *
 * The code is organized as follows:
 *
 * 1. Asynchronous copy mechanisms that wrap the PTX instructions
 *
 * 2. Device memcpy_async implementation per copy direction (global to shared,
 *    shared to global, etc). Dispatches to fastest mechanism based on requested
 *    completion mechanism(s), pointer alignment, and architecture.
 *
 * 3. Host and device memcpy_async implementations. Host implementation is
 *    basically a memcpy wrapper; device implementation dispatches based on the
 *    direction of the copy.
 *
 * 4. __memcpy_async_barrier:
 *    a) Sets the allowed completion mechanisms based on the barrier location
 *    b) Calls the host or device memcpy_async implementation
 *    c) If necessary, synchronizes with the barrier based on the returned
 *    completion mechanism.
 *
 * 5. The public memcpy_async function overloads. Call into
 *    __memcpy_async_barrier.
 *
 ***********************************************************************/

/***********************************************************************
 * Asynchronous copy mechanisms:
 *
 * 1. cp.async.bulk: shared  <- global
 * 2. TODO: cp.async.bulk: cluster <- shared
 * 3. TODO: cp.async.bulk: global  <- shared
 * 4. cp.async:      shared  <- global
 * 5. normal synchronous copy (fallback)
 ***********************************************************************/

#  if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();
template <typename _Group>
inline __device__ void
__cp_async_bulk_shared_global(const _Group& __g, char* __dest, const char* __src, size_t __size, uint64_t* __bar_handle)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                    (if (__g.thread_rank() == 0) {
                      _CUDA_VPTX::cp_async_bulk(
                        _CUDA_VPTX::space_cluster, _CUDA_VPTX::space_global, __dest, __src, __size, __bar_handle);
                    }),
                    (__cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();));
}
#  endif // __cccl_ptx_isa >= 800

extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();
template <size_t _Copy_size>
inline __device__ void __cp_async_shared_global(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

  // If `if constexpr` is not available, this function gets instantiated even
  // if is not called. Do not static_assert in that case.
#  if _CCCL_STD_VER >= 2017
  static_assert(_Copy_size == 4 || _Copy_size == 8 || _Copy_size == 16,
                "cp.async.shared.global requires a copy size of 4, 8, or 16.");
#  endif // _CCCL_STD_VER >= 2017

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %2;"
                  :
                  : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__dest))),
                    "l"(static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__src))),
                    "n"(_Copy_size)
                  : "memory");),
    (__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}

template <>
inline __device__ void __cp_async_shared_global<16>(char* __dest, const char* __src)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
  // When copying 16 bytes, it is possible to skip L1 cache (.cg).
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;"
                  :
                  : "r"(static_cast<_CUDA_VSTD::uint32_t>(__cvta_generic_to_shared(__dest))),
                    "l"(static_cast<_CUDA_VSTD::uint64_t>(__cvta_generic_to_global(__src))),
                    "n"(16)
                  : "memory");),
    (__cuda_ptx_cp_async_shared_global_is_not_supported_before_SM_80__();));
}

template <size_t _Alignment, typename _Group>
inline __device__ void
__cp_async_shared_global_mechanism(_Group __g, char* __dest, const char* __src, _CUDA_VSTD::size_t __size)
{
  // If `if constexpr` is not available, this function gets instantiated even
  // if is not called. Do not static_assert in that case.
#  if _CCCL_STD_VER >= 2017
  static_assert(4 <= _Alignment, "cp.async requires at least 4-byte alignment");
#  endif // _CCCL_STD_VER >= 2017

  // Maximal copy size is 16.
  constexpr int __copy_size = (_Alignment > 16) ? 16 : _Alignment;
  // We use an int offset here, because we are copying to shared memory,
  // which is easily addressable using int.
  const int __group_size = __g.size();
  const int __group_rank = __g.thread_rank();
  const int __stride     = __group_size * __copy_size;
  for (int __offset = __group_rank * __copy_size; __offset < static_cast<int>(__size); __offset += __stride)
  {
    __cp_async_shared_global<__copy_size>(__dest + __offset, __src + __offset);
  }
}

template <size_t _Copy_size>
struct __copy_chunk
{
  _CCCL_ALIGNAS(_Copy_size) char data[_Copy_size];
};

template <size_t _Alignment, typename _Group>
inline __host__ __device__ void
__cp_async_fallback_mechanism(_Group __g, char* __dest, const char* __src, _CUDA_VSTD::size_t __size)
{
  // Maximal copy size is 16 bytes
  constexpr _CUDA_VSTD::size_t __copy_size = (_Alignment > 16) ? 16 : _Alignment;
  using __chunk_t                          = __copy_chunk<__copy_size>;

  // "Group"-strided loop over memory
  const size_t __stride = __g.size() * __copy_size;

  // An unroll factor of 64 ought to be enough for anybody. This unroll pragma
  // is mainly intended to place an upper bound on loop unrolling. The number
  // is more than high enough for the intended use case: an unroll factor of
  // 64 allows moving 4 * 64 * 256 = 64kb in one unrolled loop with 256
  // threads (copying ints). On the other hand, in the unfortunate case that
  // we have to move 1024 bytes / thread with char width, then we prevent
  // fully unrolling the loop to 1024 copy instructions. This prevents the
  // compile times from increasing unreasonably, and also has neglibible
  // impact on runtime performance.
  _LIBCUDACXX_PRAGMA_UNROLL(64)
  for (_CUDA_VSTD::size_t __offset = __g.thread_rank() * __copy_size; __offset < __size; __offset += __stride)
  {
    __chunk_t tmp                                    = *reinterpret_cast<const __chunk_t*>(__src + __offset);
    *reinterpret_cast<__chunk_t*>(__dest + __offset) = tmp;
  }
}

/***********************************************************************
 * cuda::memcpy_async dispatch helper functions
 *
 * - __get_size_align struct to determine the alignment from a size type.
 ***********************************************************************/

// The __get_size_align struct provides a way to query the guaranteed
// "alignment" of a provided size. In this case, an n-byte aligned size means
// that the size is a multiple of n.
//
// Use as follows:
// static_assert(__get_size_align<size_t>::align == 1)
// static_assert(__get_size_align<aligned_size_t<n>>::align == n)

// Default impl: always returns 1.
template <typename, typename = void>
struct __get_size_align
{
  static constexpr int align = 1;
};

// aligned_size_t<n> overload: return n.
template <typename T>
struct __get_size_align<T, _CUDA_VSTD::void_t<decltype(T::align)>>
{
  static constexpr int align = T::align;
};

/***********************************************************************
 * cuda::memcpy_async dispatch
 *
 * The dispatch mechanism takes all the arguments and dispatches to the
 * fastest asynchronous copy mechanism available.
 *
 * It returns a __completion_mechanism that indicates which completion mechanism
 * was used by the copy mechanism. This value can be used by the sync object to
 * further synchronize if necessary.
 *
 ***********************************************************************/

template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _CCCL_DEVICE inline __completion_mechanism __dispatch_memcpy_async_any_to_any(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  uint32_t __allowed_completions,
  uint64_t* __bar_handle)
{
  __cp_async_fallback_mechanism<_Align>(__group, __dest_char, __src_char, __size);
  return __completion_mechanism::__sync;
}

template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _CCCL_DEVICE inline __completion_mechanism __dispatch_memcpy_async_global_to_shared(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  uint32_t __allowed_completions,
  uint64_t* __bar_handle)
{
#  if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (const bool __can_use_complete_tx = __allowed_completions & uint32_t(__completion_mechanism::__mbarrier_complete_tx);
     _LIBCUDACXX_UNUSED_VAR(__can_use_complete_tx);
     _CCCL_ASSERT(__can_use_complete_tx == (nullptr != __bar_handle),
                  "Pass non-null bar_handle if and only if can_use_complete_tx.");
     _CCCL_IF_CONSTEXPR (_Align >= 16) {
       if (__can_use_complete_tx && __isShared(__bar_handle))
       {
         __cp_async_bulk_shared_global(__group, __dest_char, __src_char, __size, __bar_handle);
         return __completion_mechanism::__mbarrier_complete_tx;
       }
     }
     // Fallthrough to SM 80..
     ));
#  endif // __cccl_ptx_isa >= 800

  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (_CCCL_IF_CONSTEXPR (_Align >= 4) {
      const bool __can_use_async_group = __allowed_completions & uint32_t(__completion_mechanism::__async_group);
      if (__can_use_async_group)
      {
        __cp_async_shared_global_mechanism<_Align>(__group, __dest_char, __src_char, __size);
        return __completion_mechanism::__async_group;
      }
    }
     // Fallthrough..
     ));

  __cp_async_fallback_mechanism<_Align>(__group, __dest_char, __src_char, __size);
  return __completion_mechanism::__sync;
}

// __dispatch_memcpy_async is the internal entry point for dispatching to the correct memcpy_async implementation.
template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __completion_mechanism __dispatch_memcpy_async(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions,
  uint64_t* __bar_handle)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (
      // Dispatch based on direction of the copy: global to shared, shared to
      // global, etc.

      // CUDA compilers <= 12.2 may not propagate assumptions about the state space
      // of pointers correctly. Therefore, we
      // 1) put the code for each copy direction in a separate function, and
      // 2) make sure none of the code paths can reach each other by "falling through".
      //
      // See nvbug 4074679 and also PR #478.
      if (__isGlobal(__src_char) && __isShared(__dest_char)) {
        return __dispatch_memcpy_async_global_to_shared<_Align>(
          __group, __dest_char, __src_char, __size, __allowed_completions, __bar_handle);
      } else {
        return __dispatch_memcpy_async_any_to_any<_Align>(
          __group, __dest_char, __src_char, __size, __allowed_completions, __bar_handle);
      }),
    (
      // Host code path:
      if (__group.thread_rank() == 0) {
        memcpy(__dest_char, __src_char, __size);
      } return __completion_mechanism::__sync;));
}

template <_CUDA_VSTD::size_t _Align, typename _Group>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI __completion_mechanism __dispatch_memcpy_async(
  _Group const& __group,
  char* __dest_char,
  char const* __src_char,
  _CUDA_VSTD::size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions)
{
  _CCCL_ASSERT(!(__allowed_completions & uint32_t(__completion_mechanism::__mbarrier_complete_tx)),
               "Cannot allow mbarrier_complete_tx completion mechanism when not passing a barrier. ");
  return __dispatch_memcpy_async<_Align>(__group, __dest_char, __src_char, __size, __allowed_completions, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

struct __single_thread_group
{
  _LIBCUDACXX_HIDE_FROM_ABI void sync() const {}
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::size_t size() const
  {
    return 1;
  };
  _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::size_t thread_rank() const
  {
    return 0;
  };
};

template <typename _Group, class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment __memcpy_async_barrier(
  _Group const& __group, _Tp* __destination, _Tp const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  static_assert(_CUDA_VSTD::is_trivially_copyable<_Tp>::value, "memcpy_async requires a trivially copyable type");

  // 1. Determine which completion mechanisms can be used with the current
  // barrier. A local shared memory barrier, i.e., block-scope barrier in local
  // shared memory, supports the mbarrier_complete_tx mechanism in addition to
  // the async group mechanism.
  _CUDA_VSTD::uint32_t __allowed_completions =
    __is_local_smem_barrier(__barrier)
      ? (_CUDA_VSTD::uint32_t(__completion_mechanism::__async_group)
         | _CUDA_VSTD::uint32_t(__completion_mechanism::__mbarrier_complete_tx))
      : _CUDA_VSTD::uint32_t(__completion_mechanism::__async_group);

  // Alignment: Use the maximum of the alignment of _Tp and that of a possible cuda::aligned_size_t.
  constexpr _CUDA_VSTD::size_t __size_align = __get_size_align<_Size>::align;
  constexpr _CUDA_VSTD::size_t __align      = (alignof(_Tp) < __size_align) ? __size_align : alignof(_Tp);
  // Cast to char pointers. We don't need the type for alignment anymore and
  // erasing the types reduces the number of instantiations of down-stream
  // functions.
  char* __dest_char      = reinterpret_cast<char*>(__destination);
  char const* __src_char = reinterpret_cast<char const*>(__source);

  // 2. Issue actual copy instructions.
  auto __bh = __try_get_barrier_handle(__barrier);
  auto __cm = __dispatch_memcpy_async<__align>(__group, __dest_char, __src_char, __size, __allowed_completions, __bh);

  // 3. Synchronize barrier with copy instructions.
  return __memcpy_completion_impl::__defer(__cm, __group, __size, __barrier);
}

template <typename _Group, class _Tp, _CUDA_VSTD::size_t _Alignment, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment memcpy_async(
  _Group const& __group,
  _Tp* __destination,
  _Tp const* __source,
  aligned_size_t<_Alignment> __size,
  barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(__group, __destination, __source, __size, __barrier);
}

template <class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment
memcpy_async(_Tp* __destination, _Tp const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(__single_thread_group{}, __destination, __source, __size, __barrier);
}

template <typename _Group, class _Tp, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment memcpy_async(
  _Group const& __group,
  _Tp* __destination,
  _Tp const* __source,
  _CUDA_VSTD::size_t __size,
  barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(__group, __destination, __source, __size, __barrier);
}

template <typename _Group, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment memcpy_async(
  _Group const& __group,
  void* __destination,
  void const* __source,
  _CUDA_VSTD::size_t __size,
  barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(
    __group, reinterpret_cast<char*>(__destination), reinterpret_cast<char const*>(__source), __size, __barrier);
}

template <typename _Group, _CUDA_VSTD::size_t _Alignment, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment memcpy_async(
  _Group const& __group,
  void* __destination,
  void const* __source,
  aligned_size_t<_Alignment> __size,
  barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(
    __group, reinterpret_cast<char*>(__destination), reinterpret_cast<char const*>(__source), __size, __barrier);
}

template <typename _Size, thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI async_contract_fulfillment
memcpy_async(void* __destination, void const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  return __memcpy_async_barrier(
    __single_thread_group{},
    reinterpret_cast<char*>(__destination),
    reinterpret_cast<char const*>(__source),
    __size,
    __barrier);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_CUDA_COMPILER

#endif // _LIBCUDACXX___CUDA_BARRIER_H
