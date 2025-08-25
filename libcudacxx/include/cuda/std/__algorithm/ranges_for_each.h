//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_H
#define _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/in_fun_result.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

template <class _Iter, class _Func>
using for_each_result = in_fun_result<_Iter, _Func>;

_CCCL_BEGIN_NAMESPACE_CPO(__for_each)
struct __fn
{
private:
  template <class _Iter, class _Sent, class _Proj, class _Func>
  _CCCL_API constexpr static for_each_result<_Iter, _Func>
  __for_each_impl(_Iter __first, _Sent __last, _Func& __func, _Proj& __proj)
  {
    for (; __first != __last; ++__first)
    {
      ::cuda::std::invoke(__func, ::cuda::std::invoke(__proj, *__first));
    }
    return {::cuda::std::move(__first), ::cuda::std::move(__func)};
  }

public:
  _CCCL_TEMPLATE(class _Iter, class _Sent, class _Func, class _Proj = identity)
  _CCCL_REQUIRES(input_iterator<_Iter> _CCCL_AND sentinel_for<_Sent, _Iter> _CCCL_AND
                   indirectly_unary_invocable<_Func, projected<_Iter, _Proj>>)
  _CCCL_API constexpr for_each_result<_Iter, _Func>
  operator()(_Iter __first, _Sent __last, _Func __func, _Proj __proj = {}) const
  {
    return __for_each_impl(::cuda::std::move(__first), ::cuda::std::move(__last), __func, __proj);
  }

  _CCCL_TEMPLATE(class _Range, class _Func, class _Proj = identity)
  _CCCL_REQUIRES(input_range<_Range> _CCCL_AND indirectly_unary_invocable<_Func, projected<iterator_t<_Range>, _Proj>>)
  _CCCL_API constexpr for_each_result<borrowed_iterator_t<_Range>, _Func>
  operator()(_Range&& __range, _Func __func, _Proj __proj = {}) const
  {
    return __for_each_impl(::cuda::std::ranges::begin(__range), ::cuda::std::ranges::end(__range), __func, __proj);
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto for_each = __for_each::__fn{};
} // namespace __cpo

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ALGORITHM_RANGES_FOR_EACH_H
