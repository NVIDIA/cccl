//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H
#define _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief __try_get_barrier_handle returns barrier handle of block-scoped barriers and a nullptr otherwise.
template <thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t* __try_get_barrier_handle(barrier<_Sco, _CompF>& __barrier)
{
  return nullptr;
}

template <>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t*
__try_get_barrier_handle<::cuda::thread_scope_block, _CUDA_VSTD::__empty_completion>(
  barrier<thread_scope_block>& __barrier)
{
  (void) __barrier;
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE, (return ::cuda::device::barrier_native_handle(__barrier);), NV_ANY_TARGET, (return nullptr;));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H
