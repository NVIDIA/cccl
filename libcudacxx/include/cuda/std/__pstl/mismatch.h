//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_MISMATCH_H
#define _CUDA_STD___PSTL_MISMATCH_H

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
#  include <cuda/std/__algorithm/mismatch.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/not_fn.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_comparable.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/tuple>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIter1, class _InputIter2, class _BinaryPred = ::cuda::std::equal_to<>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIter1> _CCCL_AND __has_forward_traversal<_InputIter2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API pair<_InputIter1, _InputIter2> mismatch(
  [[maybe_unused]] const _Policy& __policy,
  _InputIter1 __first1,
  _InputIter1 __last1,
  _InputIter2 __first2,
  _BinaryPred __pred = {})
{
  static_assert(indirect_binary_predicate<_BinaryPred, _InputIter1, _InputIter2>,
                "cuda::std::mismatch: BinaryPred must satisfy "
                "indirect_binary_predicate<BinaryPred, InputIter1, InputIter2>");

  if (__first1 == __last1)
  {
    return pair<_InputIter1, _InputIter2>{__first1, __first2};
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::mismatch");
    const auto __count = ::cuda::std::distance(__first1, __last1);
    auto __zip_first   = ::cuda::zip_iterator{::cuda::std::move(__first1), ::cuda::std::move(__first2)};
    auto __zip_last    = __zip_first + __count;
    auto __result      = __dispatch(
      __policy,
      ::cuda::std::move(__zip_first),
      ::cuda::std::move(__zip_last),
      ::cuda::zip_function{::cuda::std::not_fn(::cuda::std::move(__pred))});
    return pair<_InputIter1, _InputIter2>{
      ::cuda::std::get<0>(__result.__iterators()), ::cuda::std::get<1>(__result.__iterators())};
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::mismatch requires at least one selected backend");
    return ::cuda::std::mismatch(
      ::cuda::std::move(__first1), ::cuda::std::move(__last1), ::cuda::std::move(__first2), ::cuda::std::move(__pred));
  }
}

_CCCL_TEMPLATE(class _Policy, class _InputIter1, class _InputIter2, class _BinaryPred = ::cuda::std::equal_to<>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIter1> _CCCL_AND __has_forward_traversal<_InputIter2> _CCCL_AND
                 is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API pair<_InputIter1, _InputIter2> mismatch(
  [[maybe_unused]] const _Policy& __policy,
  _InputIter1 __first1,
  _InputIter1 __last1,
  _InputIter2 __first2,
  _InputIter2 __last2,
  _BinaryPred __pred = {})
{
  static_assert(indirect_binary_predicate<_BinaryPred, _InputIter1, _InputIter2>,
                "cuda::std::mismatch: BinaryPred must satisfy "
                "indirect_binary_predicate<BinaryPred, InputIter1, InputIter2>");

  // Different than equal, if either range is empty we return {first1, first2}
  if (__first1 == __last1 || __first2 == __last2)
  {
    return pair<_InputIter1, _InputIter2>{__first1, __first2};
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::mismatch");
    auto __result = __dispatch(
      __policy,
      ::cuda::zip_iterator{::cuda::std::move(__first1), ::cuda::std::move(__first2)},
      ::cuda::zip_iterator{::cuda::std::move(__last1), ::cuda::std::move(__last2)},
      ::cuda::zip_function{::cuda::std::not_fn(::cuda::std::move(__pred))});
    return pair<_InputIter1, _InputIter2>{
      ::cuda::std::get<0>(__result.__iterators()), ::cuda::std::get<1>(__result.__iterators())};
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::mismatch requires at least one selected backend");
    return ::cuda::std::mismatch(
      ::cuda::std::move(__first1),
      ::cuda::std::move(__last1),
      ::cuda::std::move(__first2),
      ::cuda::std::move(__last2),
      ::cuda::std::move(__pred));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_MISMATCH_H
