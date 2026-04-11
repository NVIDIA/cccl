//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_PARTITION_COPY_H
#define _CUDA_STD___PSTL_CUDA_PARTITION_COPY_H

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

#  include <cub/device/device_partition.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/partition_copy.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__partition_copy, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPred>
  [[nodiscard]] _CCCL_HOST_API static size_t __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator1 __result_true,
    _OutputIterator2 __result_false,
    _UnaryPred __pred)
  {
    using _OffsetType = size_t;
    using __output_wrapper_t =
      CUB_NS_QUALIFIER::detail::select::partition_distinct_output_t<_OutputIterator1, _OutputIterator2>;
    __output_wrapper_t __result{::cuda::std::move(__result_true), ::cuda::std::move(__result_false)};

    _OffsetType __ret;
    const auto __count = static_cast<_OffsetType>(::cuda::std::distance(__first, __last));

    // Determine temporary device storage requirements for device_partition
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DevicePartition::If,
      "__pstl_cuda_partition_copy: determination of device storage for cub::DevicePartition::If failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      __result,
      static_cast<_OffsetType*>(nullptr),
      __count,
      __pred,
      0);

    // Allocate memory for result
    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    {
      __temporary_storage<_OffsetType> __storage{__policy, __num_bytes, 1};

      // Run the kernel, the standard requires that the input and output range do not overlap
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DevicePartition::If,
        "__pstl_cuda_partition_copy: kernel launch of cub::DevicePartition::If failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        ::cuda::std::move(__result),
        __storage.template __get_ptr<0>(),
        __count,
        ::cuda::std::move(__pred),
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_partition_copy: copy of result from device to host failed",
        ::cuda::std::addressof(__ret),
        __storage.template __get_ptr<0>(),
        sizeof(_OffsetType),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();
    return static_cast<size_t>(__ret);
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator1, class _OutputIterator2, class _UnaryPred)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator1> _CCCL_AND
                   __has_forward_traversal<_OutputIterator2>)
  [[nodiscard]] _CCCL_HOST_API size_t operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator1 __result_true,
    _OutputIterator2 __result_false,
    _UnaryPred __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator1>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator2>)
    {
      try
      {
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          ::cuda::std::move(__last),
          ::cuda::std::move(__result_true),
          ::cuda::std::move(__result_false),
          ::cuda::std::move(__pred));
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
                    "CUDA backend of cuda::std::partition_copy requires random access iterators");
      return ::cuda::std::partition_copy(
        ::cuda::std::move(__first),
        ::cuda::std::move(__last),
        ::cuda::std::move(__result_true),
        ::cuda::std::move(__result_false),
        ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_PARTITION_COPY_H
