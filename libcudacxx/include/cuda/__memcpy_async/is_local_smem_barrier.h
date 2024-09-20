//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H
#define _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__type_traits/is_same.h>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief __is_local_smem_barrier returns true if barrier is (1) block-scoped and (2) located in shared memory.
template <thread_scope _Sco,
          typename _CompF,
          bool _Is_mbarrier = (_Sco == thread_scope_block)
                           && _CCCL_TRAIT(_CUDA_VSTD::is_same, _CompF, _CUDA_VSTD::__empty_completion)>
_LIBCUDACXX_HIDE_FROM_ABI bool __is_local_smem_barrier(barrier<_Sco, _CompF>& __barrier)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return _Is_mbarrier && __isShared(&__barrier);), (return false;));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H
