//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_H
#define _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/ranges_operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/projected.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/dangling.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__find_if)
struct __fn
{
  template <class _Ip, class _Sp, class _Pred, class _Proj>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Ip __find_if_impl(_Ip __first, _Sp __last, _Pred& __pred, _Proj& __proj)
  {
    for (; __first != __last; ++__first)
    {
      if (_CUDA_VSTD::invoke(__pred, _CUDA_VSTD::invoke(__proj, *__first)))
      {
        break;
      }
    }
    return __first;
  }

  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp, class _Pred, class _Proj = identity)
  _LIBCUDACXX_REQUIRES(input_iterator<_Ip> _LIBCUDACXX_AND sentinel_for<_Sp, _Ip> _LIBCUDACXX_AND
                         indirect_unary_predicate<_Pred, projected<_Ip, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Ip
  operator()(_Ip __first, _Sp __last, _Pred __pred, _Proj __proj = {}) const
  {
    return __find_if_impl(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __pred, __proj);
  }

  _LIBCUDACXX_TEMPLATE(class _Rp, class _Pred, class _Proj = identity)
  _LIBCUDACXX_REQUIRES(
    input_range<_Rp> _LIBCUDACXX_AND indirect_unary_predicate<_Pred, projected<iterator_t<_Rp>, _Proj>>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr borrowed_iterator_t<_Rp>
  operator()(_Rp&& __r, _Pred __pred, _Proj __proj = {}) const
  {
    return __find_if_impl(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r), __pred, __proj);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto find_if = __find_if::__fn{};
} // namespace __cpo
_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

#endif // _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_H
