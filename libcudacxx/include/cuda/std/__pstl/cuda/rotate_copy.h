//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_ROTATE_COPY_H
#define _CUDA_STD___PSTL_CUDA_ROTATE_COPY_H

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
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/rotate_copy.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

struct __rotate_copy_fn
{
  size_t __middle_;

  [[nodiscard]] _CCCL_API constexpr bool operator()(size_t __index) const noexcept
  {
    return __index >= __middle_;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__rotate_copy, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _OutputIterator>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __middle,
    _InputIterator __last,
    _OutputIterator __result)
  {
    using _OffsetType = size_t;

    const auto __count  = static_cast<_OffsetType>(::cuda::std::distance(__first, __last));
    const auto __count1 = static_cast<_OffsetType>(::cuda::std::distance(__first, __middle));

    // Knowing the sizes of the partitions, we can directly write into them
    using __output_wrapper_t =
      CUB_NS_QUALIFIER::detail::select::partition_distinct_output_t<_OutputIterator, _OutputIterator>;
    __output_wrapper_t __output_wrapper{
      __result, __result + static_cast<iter_difference_t<_OutputIterator>>(__count - __count1)};

    // Determine temporary device storage requirements for cub::DevicePartition::Flagged
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DevicePartition::Flagged,
      "__pstl_cuda_rotate_copy: determination of device storage for cub::DevicePartition::Flagged failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      ::cuda::transform_iterator{::cuda::counting_iterator<size_t>{0}, __rotate_copy_fn{__count1}},
      __output_wrapper,
      static_cast<_OffsetType*>(nullptr),
      __count,
      nullptr);

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);

    {
      // Allocate memory for result
      __temporary_storage<_OffsetType> __storage{__policy, __num_bytes, 1};

      // Run the kernel, we use the flagged kernel because we know the exact ordering we want
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DevicePartition::Flagged,
        "__pstl_cuda_rotate_copy: kernel launch of cub::DevicePartition::Flagged failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        ::cuda::transform_iterator{::cuda::counting_iterator<size_t>{0}, __rotate_copy_fn{__count1}},
        ::cuda::std::move(__output_wrapper),
        __storage.template __get_ptr<0>(),
        __count,
        __stream.get());
    }

    __stream.sync();
    return __result + static_cast<iter_difference_t<_OutputIterator>>(__count);
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __middle,
    _InputIterator __last,
    _OutputIterator __result) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          ::cuda::std::move(__middle),
          ::cuda::std::move(__last),
          ::cuda::std::move(__result));
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
                    "CUDA backend of cuda::std::rotate_copy requires random access iterators");
      return ::cuda::std::rotate_copy(
        ::cuda::std::move(__first), ::cuda::std::move(__middle), ::cuda::std::move(__last), ::cuda::std::move(__result));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_ROTATE_COPY_H
