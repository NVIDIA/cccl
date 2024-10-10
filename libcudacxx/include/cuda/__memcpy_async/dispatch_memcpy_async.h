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

#ifndef _CUDA_PTX__MEMCPY_ASYNC_DISPATCH_MEMCPY_ASYNC_H_
#define _CUDA_PTX__MEMCPY_ASYNC_DISPATCH_MEMCPY_ASYNC_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memcpy_async/completion_mechanism.h>
#include <cuda/__memcpy_async/cp_async_bulk_shared_global.h>
#include <cuda/__memcpy_async/cp_async_fallback.h>
#include <cuda/__memcpy_async/cp_async_shared_global.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/detail/libcxx/include/cstring>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

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
  _CUDA_VSTD::uint32_t __allowed_completions,
  _CUDA_VSTD::uint64_t* __bar_handle)
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
  _CUDA_VSTD::uint32_t __allowed_completions,
  _CUDA_VSTD::uint64_t* __bar_handle)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (const bool __can_use_complete_tx = __allowed_completions & uint32_t(__completion_mechanism::__mbarrier_complete_tx);
     (void) __can_use_complete_tx;
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
#endif // __cccl_ptx_isa >= 800

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
  _CUDA_VSTD::size_t __size,
  _CUDA_VSTD::uint32_t __allowed_completions,
  _CUDA_VSTD::uint64_t* __bar_handle)
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

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA_PTX__MEMCPY_ASYNC_DISPATCH_MEMCPY_ASYNC_H_
