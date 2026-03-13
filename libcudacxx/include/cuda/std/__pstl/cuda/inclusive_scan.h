//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_INCLUSIVE_SCAN_H
#define _CUDA_STD___PSTL_CUDA_INCLUSIVE_SCAN_H

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

#  include <cub/device/device_scan.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/tabulate_output_iterator.h>
#  include <cuda/__memory_pool/device_memory_pool.h>
#  include <cuda/__memory_resource/get_memory_resource.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__numeric/inclusive_scan.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__inclusive_scan, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    iter_difference_t<_InputIterator> __count,
    _OutputIterator __result,
    _BinaryOp __binary_op,
    _Tp __init)
  {
    _OutputIterator __ret = __result + iter_difference_t<_OutputIterator>(__count);

    // Determine temporary device storage requirements for reduce
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceScan::InclusiveScanInit,
      "__pstl_cuda_inclusive_scan: determination of device storage for cub::DeviceScan::InclusiveScanInit failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      __result,
      __binary_op,
      __init,
      __count);

    // Allocate memory for result
    auto __stream   = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    auto __resource = ::cuda::__call_or(
      ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__stream.device()), __policy);

    {
      __temporary_storage<decltype(__resource)> __storage{__stream, __resource, __num_bytes};

      // Run the scan
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceScan::InclusiveScanInit,
        "__pstl_cuda_exclusive_scan: kernel launch of cub::DeviceScan::InclusiveScanInit failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        ::cuda::std::move(__result),
        ::cuda::std::move(__binary_op),
        __init,
        __count,
        __stream.get());
    }

    __stream.sync();
    return __ret;
  }

  template <class _Policy, class _InputIterator, class _OutputIterator, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    iter_difference_t<_InputIterator> __count,
    _OutputIterator __result,
    _BinaryOp __binary_op)
  {
    _OutputIterator __ret = __result + iter_difference_t<_OutputIterator>(__count);

    // Determine temporary device storage requirements for reduce
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceScan::InclusiveScan,
      "__pstl_cuda_inclusive_scan: determination of device storage for cub::DeviceScan::InclusiveScan failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      __result,
      __binary_op,
      __count);

    // Allocate memory for result
    auto __stream   = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    auto __resource = ::cuda::__call_or(
      ::cuda::mr::get_memory_resource, ::cuda::device_default_memory_pool(__stream.device()), __policy);

    {
      __temporary_storage<decltype(__resource)> __storage{__stream, __resource, __num_bytes};

      // Run the scan
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceScan::InclusiveScan,
        "__pstl_cuda_exclusive_scan: kernel launch of cub::DeviceScan::InclusiveScan failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        ::cuda::std::move(__result),
        ::cuda::std::move(__binary_op),
        __count,
        __stream.get());
    }

    __stream.sync();
    return __ret;
  }

  template <class _Policy, class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _BinaryOp __binary_op,
    _Tp __init) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        const auto __count = ::cuda::std::distance(__first, __last);
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          __count,
          ::cuda::std::move(__result),
          ::cuda::std::move(__binary_op),
          __init);
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
                    "__pstl_dispatch: CUDA backend of cuda::std::inclusive_scan requires at least random access "
                    "iterators");
      return ::cuda::std::inclusive_scan(
        ::cuda::std::move(__first),
        ::cuda::std::move(__last),
        ::cuda::std::move(__result),
        ::cuda::std::move(__binary_op),
        __init);
    }
  }

  template <class _Policy, class _InputIterator, class _OutputIterator, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _BinaryOp __binary_op) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        const auto __count = ::cuda::std::distance(__first, __last);
        return __par_impl(
          __policy, ::cuda::std::move(__first), __count, ::cuda::std::move(__result), ::cuda::std::move(__binary_op));
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
                    "__pstl_dispatch: CUDA backend of cuda::std::inclusive_scan requires at least random access "
                    "iterators");
      return ::cuda::std::inclusive_scan(
        ::cuda::std::move(__first),
        ::cuda::std::move(__last),
        ::cuda::std::move(__result),
        ::cuda::std::move(__binary_op));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_INCLUSIVE_SCAN_H
