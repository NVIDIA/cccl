//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_FIND_IF_NOT_H
#define _CUDA_STD___ALGORITHM_RANGES_FIND_IF_NOT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/ranges_find_if.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

_CCCL_BEGIN_NAMESPACE_CPO(__find_if_not)
template <class _Pred>
struct __not_pred
{
  _CCCL_API constexpr __not_pred(_Pred& __pred) noexcept
      : __pred_(__pred)
  {}

  _Pred& __pred_;

  template <class _Tp>
  _CCCL_API constexpr auto operator()(_Tp&& __e) const
  {
    return !::cuda::std::invoke(__pred_, ::cuda::std::forward<_Tp>(__e));
  }
};

struct __fn
{
  _CCCL_TEMPLATE(class _Ip, class _Sp, class _Pred, class _Proj = identity)
  _CCCL_REQUIRES(input_iterator<_Ip> _CCCL_AND sentinel_for<_Sp, _Ip> _CCCL_AND
                   indirect_unary_predicate<_Pred, projected<_Ip, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr _Ip operator()(_Ip __first, _Sp __last, _Pred __pred, _Proj __proj = {}) const
  {
    return ::cuda::std::ranges::find_if(
      ::cuda::std::move(__first), ::cuda::std::move(__last), __not_pred{__pred}, ::cuda::std::move(__proj));
  }

  _CCCL_TEMPLATE(class _Rp, class _Pred, class _Proj = identity)
  _CCCL_REQUIRES(input_range<_Rp> _CCCL_AND indirect_unary_predicate<_Pred, projected<iterator_t<_Rp>, _Proj>>)
  [[nodiscard]] _CCCL_API constexpr borrowed_iterator_t<_Rp> operator()(_Rp&& __r, _Pred __pred, _Proj __proj = {}) const
  {
    return ::cuda::std::ranges::find_if(::cuda::std::forward<_Rp>(__r), __not_pred{__pred}, ::cuda::std::move(__proj));
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto find_if_not = __find_if_not::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_FIND_IF_NOT_H
