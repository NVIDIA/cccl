//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_ADJACENT_DIFFERENCE_H
#define _CUDA_STD___PSTL_CUDA_ADJACENT_DIFFERENCE_H

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

#  include <cub/device/device_adjacent_difference.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__numeric/adjacent_difference.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__adjacent_difference, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _OutputIterator, class _BinaryOp>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _BinaryOp __binary_op)
  {
    auto __count = ::cuda::std::distance(__first, __last);

    // We pass the policy as an environment to DeviceAdjacentDifference
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceAdjacentDifference::SubtractLeftCopy,
      "__pstl_cuda_merge: kernel launch of cub::DeviceAdjacentDifference::SubtractLeftCopy failed",
      ::cuda::std::move(__first),
      __result,
      __count,
      ::cuda::std::move(__binary_op),
      __policy);

    // Get the stream for synchronization after the algorithm is run
    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);
    __stream.sync();

    return __result + __count;
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator, class _BinaryOp)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
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
        return __par_impl(
          __policy,
          ::cuda::std::move(__first),
          ::cuda::std::move(__last),
          ::cuda::std::move(__result),
          ::cuda::std::move(__binary_op));
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
      return ::cuda::std::adjacent_difference(
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

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_ADJACENT_DIFFERENCE_H
