//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_IS_HEAP_H
#define _CUDA_STD___PSTL_IS_HEAP_H

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
#  include <cuda/__iterator/zip_function.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/is_heap.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__pstl/is_heap_until.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _RandomAccessIterator, class _Compare = less<>)
_CCCL_REQUIRES(__has_random_access_traversal<_RandomAccessIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API bool is_heap(
  [[maybe_unused]] const _Policy& __policy,
  _RandomAccessIterator __first,
  _RandomAccessIterator __last,
  _Compare __comp = {})
{
  static_assert(indirect_binary_predicate<_Compare, _RandomAccessIterator, _RandomAccessIterator>,
                "cuda::std::is_heap: Compare must satisfy "
                "indirect_binary_predicate<Compare, RandomAccessIterator, RandomAccessIterator>");

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::is_heap");

    using __diff_t = iter_difference_t<_RandomAccessIterator>;
    const auto __n = ::cuda::std::distance(__first, __last);
    if (__n < __diff_t(2))
    {
      return true;
    }

    // Find the first heap-property violation in the index range [1, n).
    // The result's dereferenced value is the violating child index, or n
    // if the range is a heap. `::cuda::std::get<1>(__result.__iterators())`
    // returns __last in the latter case.
    const auto __result = __dispatch(
      __policy,
      ::cuda::zip_iterator{::cuda::counting_iterator{__diff_t(1)}, __first + 1},
      ::cuda::zip_iterator{::cuda::counting_iterator{__n}, __last},
      ::cuda::zip_function{__is_heap_until_fn<_RandomAccessIterator, _Compare>{__first, ::cuda::std::move(__comp)}});
    return ::cuda::std::get<1>(__result.__iterators()) == __last;
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::is_heap requires at least one selected backend");
    return ::cuda::std::is_heap(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__comp));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_IS_HEAP_H
