//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMCPY_ASYNC_IS_LOCAL_SMEM_BARRIER_H
#define _CUDA___MEMCPY_ASYNC_IS_LOCAL_SMEM_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier.h>
#include <cuda/__memory/address_space.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__type_traits/is_same.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief __is_local_smem_barrier returns true if barrier is (1) block-scoped and (2) located in shared memory.
template <thread_scope _Sco,
          typename _CompF,
          bool _Is_mbarrier = (_Sco == thread_scope_block)
                           && ::cuda::std::is_same_v<_CompF, ::cuda::std::__empty_completion>>
_CCCL_API inline bool __is_local_smem_barrier([[maybe_unused]] barrier<_Sco, _CompF>& __barrier)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (return _Is_mbarrier && ::cuda::device::is_object_from(__barrier, ::cuda::device::address_space::shared);),
    (return false;));
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMCPY_ASYNC_IS_LOCAL_SMEM_BARRIER_H
