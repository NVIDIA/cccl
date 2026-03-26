//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_MERGE_H
#define _CUDA_STD___PSTL_MERGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/copy.h>
#  include <cuda/std/__algorithm/merge.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/copy.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/merge.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 __has_forward_traversal<_OutputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API _OutputIterator merge(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result,
  _Compare __comp)
{
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__merge, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::merge");

    if (__first1 == __last1)
    {
      return ::cuda::std::copy(
        __policy, ::cuda::std::move(__first2), ::cuda::std::move(__last2), ::cuda::std::move(__result));
    }
    else if (__first2 == __last2)
    {
      return ::cuda::std::copy(
        __policy, ::cuda::std::move(__first1), ::cuda::std::move(__last1), ::cuda::std::move(__result));
    }

    return __dispatch(
      __policy,
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__last2),
      ::cuda::std::move(__result),
      ::cuda::std::move(__comp));
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::merge requires at least one selected backend");
    return ::cuda::std::merge(
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__last2),
      ::cuda::std::move(__result),
      ::cuda::std::move(__comp));
  }
}

_CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2, class _OutputIterator)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 __has_forward_traversal<_OutputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API _OutputIterator merge(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2,
  _OutputIterator __result)
{
  return ::cuda::std::merge(
    __policy,
    ::cuda::std::move(__first1),
    ::cuda::std::move(__last1),
    ::cuda::std::move(__first2),
    ::cuda::std::move(__last2),
    ::cuda::std::move(__result),
    ::cuda::std::less{});
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_MERGE_H
