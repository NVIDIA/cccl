//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_REVERSE_H
#define _CUDA_STD___PSTL_REVERSE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/iter_swap.h>
#  include <cuda/std/__algorithm/reverse.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__iterator/reverse_iterator.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/for_each_n.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _InputIterator>
struct __reverse_fn
{
  _InputIterator __first_;
  ::cuda::std::reverse_iterator<_InputIterator> __last_;
  iter_difference_t<_InputIterator> __count_;

  _CCCL_HOST_API constexpr __reverse_fn(
    _InputIterator __first,
    _InputIterator __last,
    iter_difference_t<_InputIterator> __count) noexcept(is_nothrow_move_constructible_v<_InputIterator>)
      : __first_(::cuda::std::move(__first))
      , __last_(::cuda::std::move(__last))
      , __count_(__count)
  {}

  _CCCL_DEVICE_API constexpr void operator()(const iter_difference_t<_InputIterator> __index) const noexcept
  {
    ::cuda::std::iter_swap(__first_ + __index, __last_ + __index);
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator)
_CCCL_REQUIRES(__has_bidirectional_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API void reverse([[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last)
{
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__for_each_n, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::reverse");

    if (__first == __last)
    {
      return;
    }

    const auto __count = ::cuda::std::distance(__first, __last);
    (void) __dispatch(__policy,
                      ::cuda::counting_iterator<iter_difference_t<_InputIterator>>{0},
                      static_cast<iter_difference_t<_InputIterator>>(__count / 2),
                      __reverse_fn{::cuda::std::move(__first), ::cuda::std::move(__last), __count});
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::reverse requires at least one selected backend");
    return ::cuda::std::reverse(::cuda::std::move(__first), ::cuda::std::move(__last));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HOSTED()

#endif // _CUDA_STD___PSTL_REVERSE_H
