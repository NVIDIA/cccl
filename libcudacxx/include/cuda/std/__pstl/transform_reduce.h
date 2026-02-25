//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_TRANSFORM_REDUCE_H
#define _CUDA_STD___PSTL_TRANSFORM_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/zip_function.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__numeric/transform_reduce.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/transform_reduce.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Tp, class _ReductionOp, class _TransformOp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API _Tp transform_reduce(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator __first,
  _InputIterator __last,
  _Tp __init,
  _ReductionOp __reduction_op,
  _TransformOp __transform_op)
{
  if (__first == __last)
  {
    return __init;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform_reduce,
                                                   _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::transform_reduce");
    const auto __count = ::cuda::std::distance(__first, __last);
    return __dispatch(
      __policy,
      ::cuda::std::move(__first),
      __count,
      ::cuda::std::move(__init),
      ::cuda::std::move(__reduction_op),
      ::cuda::std::move(__transform_op));
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::transform_reduce requires at least one selected backend");
    return ::cuda::std::transform_reduce(
      ::cuda::std::move(__first),
      ::cuda::std::move(__last),
      ::cuda::std::move(__init),
      ::cuda::std::move(__reduction_op),
      ::cuda::std::move(__transform_op));
  }
}

_CCCL_TEMPLATE(
  class _Policy, class _InputIterator1, class _InputIterator2, class _Tp, class _ReductionOp, class _TransformOp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
_CCCL_HOST_API _Tp transform_reduce(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _Tp __init,
  _ReductionOp __reduction_op,
  _TransformOp __transform_op)
{
  if (__first1 == __last1)
  {
    return __init;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform_reduce,
                                                   _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    const auto __count = ::cuda::std::distance(__first1, __last1);
    return __dispatch(
      __policy,
      ::cuda::zip_iterator{::cuda::std::move(__first1), ::cuda::std::move(__first2)},
      __count,
      ::cuda::std::move(__init),
      ::cuda::std::move(__reduction_op),
      ::cuda::zip_function{::cuda::std::move(__transform_op)});
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::transform_reduce requires at least one selected backend");
    return ::cuda::std::transform_reduce(
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__init),
      ::cuda::std::move(__reduction_op),
      ::cuda::std::move(__transform_op));
  }
}

_CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2, class _Tp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
_CCCL_HOST_API _Tp transform_reduce(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _Tp __init)
{
  return ::cuda::std::transform_reduce(
    __policy,
    ::cuda::std::move(__first1),
    ::cuda::std::move(__last1),
    ::cuda::std::move(__first2),
    ::cuda::std::move(__init),
    ::cuda::std::plus<>{},
    ::cuda::std::multiplies<>{});
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_TRANSFORM_REDUCE_H
