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

#ifndef _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_
#define _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_CCCL_CUDA_COMPILER)
#  if __cccl_ptx_isa >= 800

#    include <cuda/__ptx/instructions/cp_async_bulk.h>
#    include <cuda/__ptx/ptx_dot_variants.h>
#    include <cuda/__ptx/ptx_helper_functions.h>
#    include <cuda/std/cstdint>

#    include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();
template <typename _Group>
inline __device__ void __cp_async_bulk_shared_global(
  const _Group& __g, char* __dest, const char* __src, _CUDA_VSTD::size_t __size, _CUDA_VSTD::uint64_t* __bar_handle)
{
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                    (if (__g.thread_rank() == 0) {
                      _CUDA_VPTX::cp_async_bulk(
                        _CUDA_VPTX::space_cluster, _CUDA_VPTX::space_global, __dest, __src, __size, __bar_handle);
                    }),
                    (__cuda_ptx_cp_async_bulk_shared_global_is_not_supported_before_SM_90__();));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILER

#endif // _CUDA_PTX__MEMCPY_ASYNC_CP_ASYNC_BULK_SHARED_GLOBAL_H_
