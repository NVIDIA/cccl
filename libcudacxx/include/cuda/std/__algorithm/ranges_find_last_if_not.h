//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H
#define _CUDA_STD___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_find_last_if.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__ranges/subrange.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CPO(__find_last_if_not)
struct __fn
{
  template <class _Pred>
  struct __not_pred
  {
    _CCCL_API constexpr __not_pred(_Pred& __pred) noexcept
        : __pred_(__pred)
    {}

    _Pred& __pred_;

    template <class _Tp>
    [[nodiscard]] _CCCL_API constexpr auto operator()(_Tp&& __e) const
    {
      return !::cuda::std::invoke(__pred_, ::cuda::std::forward<_Tp>(__e));
    }
  };

  _CCCL_TEMPLATE(class _Iter, class _Sp, class _Pred, class _Proj = identity)
  _CCCL_REQUIRES(forward_iterator<_Iter> _CCCL_AND sentinel_for<_Sp, _Iter> _CCCL_AND
                   indirect_unary_predicate<_Pred, projected<_Iter, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr subrange<_Iter>
  _CCCL_STATIC_CALL_OPERATOR(_Iter __first, _Sp __last, _Pred __pred, _Proj __proj = {})
  {
    __not_pred<_Pred> __negated{__pred};
    return __find_last_if::__fn::__find_last_if_impl(
      ::cuda::std::move(__first), ::cuda::std::move(__last), __negated, __proj);
  }

  _CCCL_TEMPLATE(class _Rp, class _Pred, class _Proj = identity)
  _CCCL_REQUIRES(forward_range<_Rp> _CCCL_AND indirect_unary_predicate<_Pred, projected<iterator_t<_Rp>, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr borrowed_subrange_t<_Rp>
  _CCCL_STATIC_CALL_OPERATOR(_Rp&& __r, _Pred __pred, _Proj __proj = {})
  {
    __not_pred<_Pred> __negated{__pred};
    return __find_last_if::__fn::__find_last_if_impl(
      ::cuda::std::ranges::begin(__r), ::cuda::std::ranges::end(__r), __negated, __proj);
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto find_last_if_not = __find_last_if_not::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_FIND_LAST_IF_NOT_H
