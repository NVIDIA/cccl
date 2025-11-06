//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_REDUCE_H
#define _CUDA_STD___PSTL_CUDA_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")

#  include <cub/device/device_reduce.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__new/bad_alloc.h>
#  include <cuda/std/__numeric/reduce.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_EXECUTION

_CCCL_BEGIN_ARCH_VISIBILITY_NAMESPACE

template <>
struct __pstl_dispatch<__pstl_algorithm::__reduce, __execution_backend::__cuda>
{
  template <class _Policy, class _Iter, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _Tp
  __par_impl(_Policy __policy, _Iter __first, _Iter __last, _Tp __init, _BinaryOp __func)
  {
    // Allocate memory for result
    _Tp* __device_ret_ptr = nullptr;
    auto __stream         = __policy.query(::cuda::get_stream);

    _CCCL_TRY_CUDA_API(
      ::cudaMallocAsync,
      "__pstl_dispatch: allocation failed",
      reinterpret_cast<void**>(&__device_ret_ptr),
      sizeof(_Tp),
      __stream.get());

    const auto __count = ::cuda::std::distance(__first, __last);
    _Tp __ret;

    _CCCL_TRY_CUDA_API(
      ::cub::DeviceReduce::Reduce,
      "__pstl_cuda_for_each: cub::DeviceReduce::Reduce failed",
      ::cuda::std::move(__first),
      __device_ret_ptr,
      __count,
      ::cuda::std::move(__func),
      ::cuda::std::move(__init),
      ::cuda::std::move(__policy));

    _CCCL_TRY_CUDA_API(
      ::cudaMemcpyAsync,
      "__pstl_cuda_for_each: copy of result from device to host failed",
      ::cuda::std::addressof(__ret),
      __device_ret_ptr,
      sizeof(_Tp),
      ::cudaMemcpyDeviceToHost,
      __stream.get());

    _CCCL_TRY_CUDA_API(::cudaFreeAsync, "__pstl_cuda_for_each: deallocate failed", __device_ret_ptr, __stream.get());

    __stream.sync();

    return __ret;
  }

  template <class _Policy, class _Iter, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API _Tp
  operator()(_Policy __policy, _Iter __first, _Iter __last, _Tp __init, _BinaryOp __func) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_Iter>)
    {
      return __par_impl(
        ::cuda::std::move(__policy),
        ::cuda::std::move(__first),
        ::cuda::std::move(__last),
        ::cuda::std::move(__init),
        ::cuda::std::move(__func));
    }
    else
    {
      return ::cuda::std::reduce(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__init), ::cuda::std::move(__func));
    }
  }
};

_CCCL_END_ARCH_VISIBILITY_NAMESPACE

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_REDUCE_H
