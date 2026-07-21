//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_IS_PARTITIONED_H
#define _CUDA_STD___PSTL_IS_PARTITIONED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()

#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/is_partitioned.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/tuple>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _UnaryPred>
struct __is_partitioned_fn
{
  _UnaryPred __pred_;

  template <class _Tuple>
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool operator()(const _Tuple& __tuple) const
  {
    const bool __pred_lhs = __pred_(::cuda::std::get<0>(__tuple));
    const bool __pred_rhs = __pred_(::cuda::std::get<1>(__tuple));

    return (!__pred_lhs && __pred_rhs);
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _UnaryPred)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API bool is_partitioned(
  [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _UnaryPred __pred)
{
  static_assert(indirect_unary_predicate<_UnaryPred, _InputIterator>,
                "cuda::std::is_partitioned: UnaryPred must satisfy indirect_unary_predicate<UnaryPred, InputIterator>");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::is_partitioned");

    if (__first == __last)
    {
      return true;
    }

    const auto __result = __dispatch(
      __policy,
      ::cuda::zip_iterator{__first, __first + 1},
      ::cuda::zip_iterator{__last, __last},
      __is_partitioned_fn<_UnaryPred>{::cuda::std::move(__pred)});
    return ::cuda::std::get<1>(__result.__iterators()) == __last;
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::is_partitioned requires at least one selected backend");
    return ::cuda::std::is_partitioned(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HOSTED()

#endif // _CUDA_STD___PSTL_IS_PARTITIONED_H
