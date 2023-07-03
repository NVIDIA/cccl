//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_NOT_H
#define _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_NOT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__algorithm/ranges_find_if.h"
#include "../__functional/identity.h"
#include "../__functional/invoke.h"
#include "../__functional/ranges_operations.h"
#include "../__iterator/concepts.h"
#include "../__iterator/projected.h"
#include "../__ranges/access.h"
#include "../__ranges/concepts.h"
#include "../__ranges/dangling.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__find_if_not)
template<class _Pred>
struct _Not_pred {
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Not_pred(_Pred& __pred) noexcept : __pred_(__pred) {}

  _Pred& __pred_;

  template<class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr auto operator()(_Tp&& __e) const {
    return !_CUDA_VSTD::invoke(__pred_, _CUDA_VSTD::forward<_Tp>(__e));
  }
};

struct __fn {
  _LIBCUDACXX_TEMPLATE(class _Ip, class _Sp, class _Pred, class _Proj = identity)
    (requires input_iterator<_Ip> _LIBCUDACXX_AND sentinel_for<_Sp, _Ip> _LIBCUDACXX_AND
              indirect_unary_predicate<_Pred, projected<_Ip, _Proj>>)
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY
  constexpr _Ip operator()(_Ip __first, _Sp __last, _Pred __pred, _Proj __proj = {}) const {
    _Not_pred __not_pred{__pred};
    return _CUDA_VRANGES::__find_if_impl(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last), __not_pred, __proj);
  }

  _LIBCUDACXX_TEMPLATE(class _Rp, class _Pred, class _Proj = identity)
    (requires input_range<_Rp> _LIBCUDACXX_AND indirect_unary_predicate<_Pred, projected<iterator_t<_Rp>, _Proj>>)
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY
  constexpr borrowed_iterator_t<_Rp> operator()(_Rp&& __r, _Pred __pred, _Proj __proj = {}) const {
    _Not_pred __not_pred{__pred};
    return _CUDA_VRANGES::__find_if_impl(_CUDA_VRANGES::begin(__r), _CUDA_VRANGES::end(__r), __not_pred, __proj);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo {
  _LIBCUDACXX_CPO_ACCESSIBILITY auto find_if_not = __find_if_not::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_RANGES

#endif // _LIBCUDACXX_STD_VER > 14

#endif // _LIBCUDACXX___ALGORITHM_RANGES_FIND_IF_NOT_H
