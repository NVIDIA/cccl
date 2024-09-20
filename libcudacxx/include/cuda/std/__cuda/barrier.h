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
#include <cuda/__barrier/barrier_thread_scope_thread.h>
#include <cuda/__barrier/completion_mechanism.h>
#include <cuda/__barrier/is_local_smem_barrier.h>
#include <cuda/__barrier/try_get_barrier_handle.h>
#include <cuda/__fwd/pipeline.h>
#include <cuda/__memcpy_async/dispatch_memcpy_async.h>
#include <cuda/__memcpy_async/memcpy_async_tx.h>
#include <cuda/__memcpy_async/memcpy_completion.h>
#include <cuda/std/__atomic/api/owned.h>
#include <cuda/std/__type_traits/void_t.h> // _CUDA_VSTD::void_t

#if defined(_CCCL_CUDA_COMPILER)
#  include <cuda/ptx> // cuda::ptx::*
#endif // _CCCL_CUDA_COMPILER

#if defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

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
