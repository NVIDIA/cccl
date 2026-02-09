//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_COUNT_H
#define _CUDA_STD___PSTL_COUNT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/std/__algorithm/count.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_comparable.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/reduce.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct __count_compare_eq
{
  _Tp __val_;

  template <class _Up>
  [[nodiscard]] _CCCL_API _CCCL_FORCEINLINE constexpr int operator()(const _Up& __rhs) const
    noexcept(__is_cpp17_nothrow_equality_comparable_v<_Tp, _Up>)
  {
    return static_cast<bool>(__val_ == __rhs) ? 1 : 0;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Tp)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API iter_difference_t<_InputIterator>
count([[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, const _Tp& __value)
{
  static_assert(__is_cpp17_equality_comparable_v<iter_reference_t<_InputIterator>, _Tp>,
                "cuda::std::count: T must be equality comparable to Iter's value type.");
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__reduce, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    const auto __count = ::cuda::std::distance(__first, __last);
    return __dispatch(
      __policy,
      ::cuda::transform_iterator{::cuda::std::move(__first), __count_compare_eq<_Tp>{__value}},
      __count,
      iter_difference_t<_InputIterator>{0},
      ::cuda::std::plus<iter_difference_t<_InputIterator>>{});
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::count requires at least one selected backend");
    return ::cuda::std::count(::cuda::std::move(__first), ::cuda::std::move(__last), __value);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_COUNT_H
