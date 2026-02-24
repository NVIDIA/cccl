//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_TRANSFORM_H
#define _CUDA_STD___PSTL_CUDA_TRANSFORM_H

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
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_transform.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/transform.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__host_stdlib/new>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/tuple>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__transform, __execution_backend::__cuda>
{
  template <class _Policy, class _OutputIterator, class _UnaryOp, class... _InputIterators, class _Predicate>
  [[nodiscard]] _CCCL_HOST_API static _OutputIterator __par_impl(
    const _Policy& __policy,
    tuple<_InputIterators...> __first,
    _OutputIterator __result,
    const int64_t __count,
    _UnaryOp __func,
    _Predicate __pred)
  {
    auto __stream    = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);
    const auto __ret = __result + __count;

    // We pass the policy as an environment to device_transform
    _CCCL_TRY_CUDA_API(
      ::cub::DeviceTransform::TransformIf,
      "cuda::std::transform: failed inside CUDA backend",
      ::cuda::std::move(__first),
      ::cuda::std::move(__result),
      __count,
      ::cuda::std::move(__pred),
      ::cuda::std::move(__func),
      __stream.get());

    __stream.sync();
    return __ret;
  }

  _CCCL_TEMPLATE(class _Policy,
                 class _InputIterator,
                 class _OutputIterator,
                 class _UnaryOp,
                 class _Predicate = CUB_NS_QUALIFIER::detail::transform::always_true_predicate)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator> _CCCL_AND
                   is_invocable_v<_UnaryOp, iter_reference_t<_InputIterator>>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _OutputIterator __result,
    _UnaryOp __func,
    _Predicate __pred = {}) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        const auto __count = ::cuda::std::distance(__first, __last);
        return __par_impl(
          __policy,
          ::cuda::std::make_tuple(::cuda::std::move(__first)),
          ::cuda::std::move(__result),
          __count,
          ::cuda::std::move(__func),
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
                    "__pstl_dispatch: CUDA backend of cuda::std::transform requires at least random access iterators");
      return ::cuda::std::transform(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), ::cuda::std::move(__func));
    }
  }

  _CCCL_TEMPLATE(class _Policy,
                 class _InputIterator1,
                 class _InputIterator2,
                 class _OutputIterator,
                 class _BinaryOp,
                 class _Predicate = CUB_NS_QUALIFIER::detail::transform::always_true_predicate)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                   __has_forward_traversal<_OutputIterator>)
  [[nodiscard]] _CCCL_HOST_API _OutputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _OutputIterator __result,
    _BinaryOp __func,
    _Predicate __pred = {}) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator1>
                  && ::cuda::std::__has_random_access_traversal<_InputIterator2>
                  && ::cuda::std::__has_random_access_traversal<_OutputIterator>)
    {
      try
      {
        const auto __count = ::cuda::std::distance(__first1, __last1);
        return __par_impl(
          __policy,
          ::cuda::std::make_tuple(::cuda::std::move(__first1), ::cuda::std::move(__first2)),
          ::cuda::std::move(__result),
          __count,
          ::cuda::std::move(__func),
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
                    "__pstl_dispatch: CUDA backend of cuda::std::transform requires at least random access iterators");
      return ::cuda::std::transform(
        ::cuda::std::move(__first1),
        ::cuda::std::move(__last1),
        ::cuda::std::move(__first2),
        ::cuda::std::move(__result),
        ::cuda::std::move(__func));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_TRANSFORM_H
