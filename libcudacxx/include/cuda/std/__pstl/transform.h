//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_TRANSFORM_H
#define _CUDA_STD___PSTL_TRANSFORM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/transform.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _OutputIterator, class _UnaryOp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND __has_forward_traversal<_OutputIterator> _CCCL_AND
                 is_execution_policy_v<_Policy>)
_CCCL_HOST_API _OutputIterator transform(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator __first,
  _InputIterator __last,
  _OutputIterator __result,
  _UnaryOp __func)
{
  static_assert(is_invocable_v<_UnaryOp, iter_reference_t<_InputIterator>>,
                "cuda::std::transform requires UnaryOp to be invocable with iter_reference_t<InputIterator>");

  static_assert(indirectly_writable<_OutputIterator, invoke_result_t<_UnaryOp, iter_reference_t<_InputIterator>>>,
                "cuda::std::transform requires OutputIterator to be indirectly writable with the return value of "
                "UnaryOp");

  if (__first == __last)
  {
    return __result;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::transform");
    return __dispatch(
      __policy,
      ::cuda::std::move(__first),
      ::cuda::std::move(__last),
      ::cuda::std::move(__result),
      ::cuda::std::move(__func));
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::transform requires at least one selected backend");
    return ::cuda::std::transform(
      ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__result), ::cuda::std::move(__func));
  }
}

_CCCL_TEMPLATE(class _Policy, class _InputIterator1, class _InputIterator2, class _OutputIterator, class _BinaryOp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator1> _CCCL_AND __has_forward_traversal<_InputIterator2> _CCCL_AND
                 __has_forward_traversal<_OutputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API _OutputIterator transform(
  [[maybe_unused]] const _Policy& __policy,
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _OutputIterator __result,
  _BinaryOp __func)
{
  static_assert(is_invocable_v<_BinaryOp, iter_reference_t<_InputIterator1>, iter_reference_t<_InputIterator2>>,
                "cuda::std::transform requires BinaryOp to be invocable with iter_reference_t<InputIterator1> and "
                "iter_reference_t<InputIterator2>");

  static_assert(
    indirectly_writable<_OutputIterator,
                        invoke_result_t<_BinaryOp, iter_reference_t<_InputIterator1>, iter_reference_t<_InputIterator2>>>,
    "cuda::std::transform requires OutputIterator to be indirectly writable with the return value of BinaryOp");

  if (__first1 == __last1)
  {
    return __result;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::transform");
    return __dispatch(
      __policy,
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__result),
      ::cuda::std::move(__func));
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::transform requires at least one selected backend");
    return ::cuda::std::transform(
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__result),
      ::cuda::std::move(__func));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_TRANSFORM_H
