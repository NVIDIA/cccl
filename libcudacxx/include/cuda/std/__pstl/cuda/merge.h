//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_MERGE_H
#define _CUDA_STD___PSTL_CUDA_MERGE_H

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

#  include <cub/device/device_merge.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/merge.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__iterator/next.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__merge, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp)
  {
    iter_difference_t<_InputIterator1> __count1 = ::cuda::std::distance(__first1, __last1);
    iter_difference_t<_InputIterator2> __count2 = ::cuda::std::distance(__first2, __last2);
    auto __ret                                  = __result + static_cast<iter_difference_t<_OutputIterator>>(__count1)
               + static_cast<iter_difference_t<_OutputIterator>>(__count2);

    // Determine temporary device storage requirements for device_merge
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
      "__pstl_cuda_merge: determination of device storage for cub::DeviceMerge::MergeKeys failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first1,
      __count1,
      __first2,
      __count2,
      __result,
      __comp);

    // Allocate memory for result
    auto __stream   = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    auto __resource = ::cuda::__call_or(
      ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__stream.device()), __policy);
    {
      __temporary_storage<decltype(__resource)> __storage{__stream, __resource, __num_bytes};

      // Run the kernel
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
        "__pstl_cuda_merge: kernel launch of cub::DeviceMerge::MergeKeys failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first1),
        __count1,
        ::cuda::std::move(__first2),
        __count2,
        ::cuda::std::move(__result),
        ::cuda::std::move(__comp),
        __stream.get());
    }

    return __ret;
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare)
  _CCCL_REQUIRES(__has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator1>
                  && ::cuda::std::__has_random_access_traversal<_InputIterator2>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        return __par_impl(
          __policy,
          ::cuda::std::move(__first1),
          ::cuda::std::move(__last1),
          ::cuda::std::move(__first2),
          ::cuda::std::move(__last2),
          ::cuda::std::move(__result),
          ::cuda::std::move(__comp));
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
      static_assert(__always_false_v<_Policy>, "CUDA backend of cuda::std::merge requires random access iterators");
      return ::cuda::std::merge(
        ::cuda::std::move(__first1),
        ::cuda::std::move(__last1),
        ::cuda::std::move(__first2),
        ::cuda::std::move(__last2),
        ::cuda::std::move(__result),
        ::cuda::std::move(__comp));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_MERGE_H
