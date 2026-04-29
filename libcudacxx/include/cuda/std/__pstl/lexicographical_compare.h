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

#  include <cuda/__iterator/zip_transform_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/lexicographical_compare.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/cstdint>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#    include <cuda/std/__pstl/cuda/transform_reduce.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Maps a pair (a, b) drawn from the two input ranges to a tri-valued state:
//   -1 -> a is strictly less than b under `comp`
//    1 -> b is strictly less than a under `comp`
//    0 -> a and b are equivalent under `comp`
// Used as the transform of a `cuda::zip_transform_iterator`, so the per-element
// state is computed lazily by the device kernels that walk the iterator.
template <class _Compare>
struct __lex_state_fn
{
  _Compare __comp_;

  _CCCL_API explicit constexpr __lex_state_fn(_Compare __comp)
      : __comp_(::cuda::std::move(__comp))
  {}

  template <class _Tp, class _Up>
  [[nodiscard]] _CCCL_DEVICE_API constexpr int8_t operator()(const _Tp& __a, const _Up& __b) const
  {
    if (__comp_(__a, __b))
    {
      return int8_t{-1};
    }
    if (__comp_(__b, __a))
    {
      return int8_t{1};
    }
    return int8_t{0};
  }
};

// Predicate handed to the find_if pass: locates the first non-equivalent pair.
struct __lex_non_zero
{
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool operator()(int8_t __state) const noexcept
  {
    return __state != int8_t{0};
  }
};

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

  [[maybe_unused]] auto __find_dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  [[maybe_unused]] auto __reduce_dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform_reduce,
                                                   _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__find_dispatch)>
                && ::cuda::std::execution::__pstl_can_dispatch<decltype(__reduce_dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::lexicographical_compare");

    if (__first1 == __last1)
    {
      return __first2 != __last2;
    }
    if (__first2 == __last2)
    {
      return false;
    }

    const auto __n1 = ::cuda::std::distance(__first1, __last1);
    const auto __n2 = ::cuda::std::distance(__first2, __last2);
    const auto __n  = ::cuda::std::min(__n1, __n2);

    // Lazy per-element state (-1 / 0 / +1) over the two input ranges, capped at
    // the common length so neither side reads past its own end.
    auto __state_first = ::cuda::zip_transform_iterator{
      __lex_state_fn<_Compare>{::cuda::std::move(__comp)}, ::cuda::std::move(__first1), ::cuda::std::move(__first2)};
    auto __state_last = __state_first + __n;

    // Pass 1 (early-terminating): index of the first non-equivalent pair.
    auto __k_iter = __find_dispatch(__policy, ::cuda::std::move(__state_first), __state_last, __lex_non_zero{});

    // No divergence within the common prefix: the shorter range is "less".
    if (__k_iter == __state_last)
    {
      return __n1 < __n2;
    }

    // Pass 2: read back the state at the divergence position via a 1-element
    // transform_reduce with identity transform and `plus<>` over `int8_t{0}`,
    // which reduces to exactly the value at `*__k_iter`.
    const int8_t __state =
      __reduce_dispatch(__policy, __k_iter, 1, int8_t{0}, ::cuda::std::plus<int8_t>{}, ::cuda::std::identity{});
    return __state < int8_t{0};
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
