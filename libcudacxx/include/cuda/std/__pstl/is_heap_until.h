//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_IS_HEAP_UNTIL_H
#define _CUDA_STD___PSTL_IS_HEAP_UNTIL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__iterator/zip_function.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/is_heap_until.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Maps a child-relative index k (0 <= k < n - 1) to the value of its parent
// in the heap, i.e. base[(k + 1 - 1) / 2] == base[k / 2]. Used as the unary
// functor of a transform_iterator to expose parent loads to the find_if
// dispatch alongside the (sequential) child loads.
template <class _RandomAccessIterator>
struct __is_heap_parent_at_fn
{
  _RandomAccessIterator __base_;

  template <class _Diff>
  [[nodiscard]] _CCCL_DEVICE_API constexpr iter_reference_t<_RandomAccessIterator> operator()(const _Diff& __k) const
  {
    return __base_[__k / _Diff(2)];
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _RandomAccessIterator, class _Compare = less<>)
_CCCL_REQUIRES(__has_random_access_traversal<_RandomAccessIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API _RandomAccessIterator is_heap_until(
  [[maybe_unused]] const _Policy& __policy,
  _RandomAccessIterator __first,
  _RandomAccessIterator __last,
  _Compare __comp = {})
{
  static_assert(indirect_binary_predicate<_Compare, _RandomAccessIterator, _RandomAccessIterator>,
                "cuda::std::is_heap_until: Compare must satisfy "
                "indirect_binary_predicate<Compare, RandomAccessIterator, RandomAccessIterator>");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::is_heap_until");

    using __diff_t = iter_difference_t<_RandomAccessIterator>;
    const auto __n = ::cuda::std::distance(__first, __last);
    if (__n < __diff_t(2))
    {
      return __last;
    }

    // Pair each child (at indices [1, n)) with its parent value, exposing both
    // loads to the find_if dispatch. comp(parent, child) holds at the first
    // heap-property violation.
    auto __parent_at = ::cuda::transform_iterator{
      ::cuda::counting_iterator{__diff_t(0)}, __is_heap_parent_at_fn<_RandomAccessIterator>{__first}};

    auto __result = __dispatch(
      __policy,
      ::cuda::zip_iterator{__parent_at, __first + 1},
      ::cuda::zip_iterator{__parent_at + (__n - 1), __last},
      ::cuda::zip_function{::cuda::std::move(__comp)});
    return ::cuda::std::get<1>(__result.__iterators());
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::is_heap_until requires at least one selected backend");
    return ::cuda::std::is_heap_until(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__comp));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_IS_HEAP_UNTIL_H
