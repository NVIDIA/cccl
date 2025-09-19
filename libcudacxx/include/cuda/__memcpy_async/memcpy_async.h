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

#ifndef _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_H_
#define _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()

#  include <cuda/__barrier/async_contract_fulfillment.h>
#  include <cuda/__barrier/barrier.h>
#  include <cuda/__barrier/barrier_block_scope.h>
#  include <cuda/__barrier/barrier_thread_scope.h>
#  include <cuda/__memcpy_async/check_preconditions.h>
#  include <cuda/__memcpy_async/memcpy_async_barrier.h>
#  include <cuda/__memory/aligned_size.h>
#  include <cuda/std/__atomic/scopes.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

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

template <typename _Group, class _Tp, ::cuda::std::size_t _Alignment, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment memcpy_async(
  _Group const& __group,
  _Tp* __destination,
  _Tp const* __source,
  aligned_size_t<_Alignment> __size,
  barrier<_Sco, _CompF>& __barrier)
{
  static_assert(_Alignment >= alignof(_Tp), "alignment must be at least the alignof(T)");
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(__group, __destination, __source, __size, __barrier);
}

template <class _Tp, typename _Size, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment
memcpy_async(_Tp* __destination, _Tp const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(__single_thread_group{}, __destination, __source, __size, __barrier);
}

template <typename _Group, class _Tp, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment memcpy_async(
  _Group const& __group,
  _Tp* __destination,
  _Tp const* __source,
  ::cuda::std::size_t __size,
  barrier<_Sco, _CompF>& __barrier)
{
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(__group, __destination, __source, __size, __barrier);
}

template <typename _Group, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment memcpy_async(
  _Group const& __group,
  void* __destination,
  void const* __source,
  ::cuda::std::size_t __size,
  barrier<_Sco, _CompF>& __barrier)
{
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(
    __group, reinterpret_cast<char*>(__destination), reinterpret_cast<char const*>(__source), __size, __barrier);
}

template <typename _Group, ::cuda::std::size_t _Alignment, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment memcpy_async(
  _Group const& __group,
  void* __destination,
  void const* __source,
  aligned_size_t<_Alignment> __size,
  barrier<_Sco, _CompF>& __barrier)
{
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(
    __group, reinterpret_cast<char*>(__destination), reinterpret_cast<char const*>(__source), __size, __barrier);
}

template <typename _Size, thread_scope _Sco, typename _CompF>
_CCCL_API inline async_contract_fulfillment
memcpy_async(void* __destination, void const* __source, _Size __size, barrier<_Sco, _CompF>& __barrier)
{
  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__destination, __source, __size), "memcpy_async preconditions unmet");
  return ::cuda::__memcpy_async_barrier(
    __single_thread_group{},
    reinterpret_cast<char*>(__destination),
    reinterpret_cast<char const*>(__source),
    __size,
    __barrier);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CUDA_COMPILER()

#endif // _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_H_
