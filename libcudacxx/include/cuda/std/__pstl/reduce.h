//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_REDUCE_H
#define _CUDA_STD___PSTL_REDUCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__iterator/readable_traits.h>
#  include <cuda/std/__numeric/reduce.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_move_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/reduce.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Iter, class _Tp, class _BinaryOp>
_CCCL_CONCEPT __indirect_binary_function = _CCCL_REQUIRES_EXPR((_Iter, _Tp, _BinaryOp))(
  requires(is_convertible_v<invoke_result_t<_BinaryOp&, iter_reference_t<_Iter>, _Tp>, _Tp>),
  requires(is_convertible_v<invoke_result_t<_BinaryOp&, _Tp, iter_reference_t<_Iter>>, _Tp>),
  requires(is_convertible_v<invoke_result_t<_BinaryOp&, _Tp, _Tp>, _Tp>),
  requires(is_convertible_v<invoke_result_t<_BinaryOp&, iter_reference_t<_Iter>, iter_reference_t<_Iter>>, _Tp>));

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _Iter, class _Tp, class _BinaryOp)
_CCCL_REQUIRES(__has_forward_traversal<_Iter> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API _Tp
reduce([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _Tp __init, _BinaryOp __func)
{
  static_assert(__indirect_binary_function<_Iter, _Tp, _BinaryOp>,
                "cuda::std::reduce: The return value of BinaryOp is not convertible to T.");
  static_assert(is_move_constructible_v<_Tp>, "cuda::std::reduce: T must be move constructible.");
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__reduce, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    return __dispatch(
      __policy,
      ::cuda::std::move(__first),
      ::cuda::std::move(__last),
      ::cuda::std::move(__init),
      ::cuda::std::move(__func));
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::reduce requires at least one selected backend");
    return ::cuda::std::reduce(
      ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__init), ::cuda::std::move(__func));
  }
}

_CCCL_TEMPLATE(class _Policy, class _Iter, class _Tp)
_CCCL_REQUIRES(__has_forward_traversal<_Iter> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API _Tp reduce(const _Policy& __policy, _Iter __first, _Iter __last, _Tp __init)
{
  return ::cuda::std::reduce(
    __policy,
    ::cuda::std::move(__first),
    ::cuda::std::move(__last),
    ::cuda::std::move(__init),
    ::cuda::std::plus<_Tp>{});
}

_CCCL_TEMPLATE(class _Policy, class _Iter)
_CCCL_REQUIRES(__has_forward_traversal<_Iter> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API iter_value_t<_Iter> reduce(const _Policy& __policy, _Iter __first, _Iter __last)
{
  return ::cuda::std::reduce(
    __policy,
    ::cuda::std::move(__first),
    ::cuda::std::move(__last),
    iter_value_t<_Iter>{},
    ::cuda::std::plus<iter_value_t<_Iter>>{});
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_REDUCE_H
