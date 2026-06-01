//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_LEXICOGRAPHICAL_COMPARE_H
#define _CUDA_STD___PSTL_LEXICOGRAPHICAL_COMPARE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()

#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/lexicographical_compare.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/lexicographical_compare.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIter1, class _InputIter2, class _Compare = ::cuda::std::less<>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIter1> _CCCL_AND __has_forward_traversal<_InputIter2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API bool lexicographical_compare(
  [[maybe_unused]] const _Policy& __policy,
  _InputIter1 __first1,
  _InputIter1 __last1,
  _InputIter2 __first2,
  _InputIter2 __last2,
  _Compare __comp = {})
{
  static_assert(indirect_strict_weak_order<_Compare, _InputIter1, _InputIter2>,
                "cuda::std::lexicographical_compare: Compare must satisfy "
                "indirect_strict_weak_order<Compare, InputIter1, InputIter2>");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__lexicographical_compare,
                                                   _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::lexicographical_compare");
    return __dispatch(
      __policy,
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__last2),
      ::cuda::std::move(__comp));
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::lexicographical_compare requires at least one selected backend");
    return ::cuda::std::lexicographical_compare(
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__last2),
      ::cuda::std::move(__comp));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HOSTED()

#endif // _CUDA_STD___PSTL_LEXICOGRAPHICAL_COMPARE_H
