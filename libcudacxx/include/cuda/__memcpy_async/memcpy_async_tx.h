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

#ifndef _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_
#define _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_

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

#    include <cuda/__barrier/aligned_size.h>
#    include <cuda/__barrier/async_contract_fulfillment.h>
#    include <cuda/__barrier/barrier_block_scope.h>
#    include <cuda/__barrier/barrier_native_handle.h>
#    include <cuda/__ptx/instructions/cp_async_bulk.h>
#    include <cuda/__ptx/ptx_dot_variants.h>
#    include <cuda/__ptx/ptx_helper_functions.h>
#    include <cuda/std/__atomic/scopes.h>
#    include <cuda/std/__type_traits/is_trivially_copyable.h>
#    include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

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

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILER

#endif // _CUDA_PTX__MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_
