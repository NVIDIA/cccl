//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_FIND_IF_H
#define _CUDA_STD___PSTL_CUDA_FIND_IF_H

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
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_find.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/find_if.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__host_stdlib/new>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__find_if, __execution_backend::__cuda>
{
  template <class _Policy, class _Iter, class _UnaryOp>
  [[nodiscard]] _CCCL_HOST_API static _Iter
  __par_impl([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _UnaryOp __pred)
  {
    const auto __num_items = ::cuda::std::distance(__first, __last);
    using __offset_type    = remove_cvref_t<decltype(__num_items)>;
    __offset_type __ret;

    // Determine temporary device storage requirements for find_if
    void* __temp_storage = nullptr;
    size_t __num_bytes   = 0;
    _CCCL_TRY_CUDA_API(
      ::cub::DeviceFind::FindIf,
      "__pstl_cuda_find_if: determining temporary storage failed",
      __temp_storage,
      __num_bytes,
      __first,
      static_cast<__offset_type*>(nullptr),
      __pred,
      __num_items);

    // Allocate memory for result
    auto __stream   = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    auto __resource = ::cuda::__call_or(
      ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__stream.device()), __policy);
    {
      __temporary_storage<__offset_type, decltype(__resource)> __storage{__stream, __resource, __num_bytes};

      // Run the find operation
      _CCCL_TRY_CUDA_API(
        ::cub::DeviceFind::FindIf,
        "__pstl_cuda_find_if: cub::DeviceFind failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        __storage.__get_result_iter(),
        ::cuda::std::move(__pred),
        __num_items,
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_find_if: copy of result from device to host failed",
        ::cuda::std::addressof(__ret),
        __storage.__res_,
        sizeof(__offset_type),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    // Need to sync before reading __ret
    __stream.sync();
    return __first + __ret;
  }

  template <class _Policy, class _Iter, class _UnaryOp>
  [[nodiscard]] _CCCL_HOST_API _Iter
  operator()([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _UnaryOp __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_Iter>)
    {
      try
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          throw __err;
        }
      }
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::find_if requires at least random access iterators");
      return ::cuda::std::find_if(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_FIND_IF_H
