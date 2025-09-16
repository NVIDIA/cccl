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

#ifndef _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_
#define _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  if __cccl_ptx_isa >= 800

#    include <cuda/__barrier/async_contract_fulfillment.h>
#    include <cuda/__barrier/barrier_block_scope.h>
#    include <cuda/__barrier/barrier_native_handle.h>
#    include <cuda/__memcpy_async/check_preconditions.h>
#    include <cuda/__memory/address_space.h>
#    include <cuda/__memory/aligned_size.h>
#    include <cuda/__ptx/instructions/cp_async_bulk.h>
#    include <cuda/__ptx/ptx_dot_variants.h>
#    include <cuda/__ptx/ptx_helper_functions.h>
#    include <cuda/std/__atomic/scopes.h>
#    include <cuda/std/__type_traits/is_trivially_copyable.h>
#    include <cuda/std/cstdint>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

extern "C" _CCCL_DEVICE void __cuda_ptx_memcpy_async_tx_is_not_supported_before_SM_90__();
template <typename _Tp, ::cuda::std::size_t _Alignment>
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
#    if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >, 4, 8)
  static_assert(::cuda::std::is_trivially_copyable<_Tp>::value, "memcpy_async_tx requires a trivially copyable type");
#    endif
  static_assert(16 <= _Alignment, "mempcy_async_tx expects arguments to be at least 16 byte aligned.");
  static_assert(_Alignment >= alignof(_Tp), "alignment must be at least the alignof(T)");

  _CCCL_ASSERT(::cuda::__memcpy_async_check_pre(__dest, __src, __size), "memcpy_async_tx preconditions unmet");

  _CCCL_ASSERT(
    ::cuda::device::is_address_from(::cuda::device::barrier_native_handle(__b), ::cuda::device::address_space::shared),
    "Barrier must be located in local shared memory.");
  _CCCL_ASSERT(::cuda::device::is_address_from(__dest, ::cuda::device::address_space::shared),
               "dest must point to shared memory.");
  _CCCL_ASSERT(::cuda::device::is_address_from(__src, ::cuda::device::address_space::global),
               "src must point to global memory.");

  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      if (::cuda::device::is_address_from(__dest, ::cuda::device::address_space::shared)
          && ::cuda::device::is_address_from(__src, ::cuda::device::address_space::global)) {
        ::cuda::ptx::cp_async_bulk(
          ::cuda::ptx::space_cluster,
          ::cuda::ptx::space_global,
          __dest,
          __src,
          static_cast<uint32_t>(__size),
          ::cuda::device::barrier_native_handle(__b));
      } else {
        // memcpy_async_tx only supports copying from global to shared
        // or from shared to remote cluster dsmem. To copy to remote
        // dsmem, we need to arrive on a cluster-scoped barrier, which
        // is not yet implemented. So we trap in this case as well.
        _CCCL_UNREACHABLE();
      }),
    (::cuda::device::__cuda_ptx_memcpy_async_tx_is_not_supported_before_SM_90__();));

  return async_contract_fulfillment::async;
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 800
#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMCPY_ASYNC_MEMCPY_ASYNC_TX_H_
