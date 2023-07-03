// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H
#define _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__concepts/constructible.h"
#include "../__concepts/derived_from.h"
#include "../__concepts/invocable.h"
#include "../__concepts/same_as.h"
#include "../__functional/compose.h"
#include "../__functional/invoke.h"
#include "../__ranges/concepts.h"
#include "../__type_traits/decay.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_nothrow_constructible.h"
#include "../__type_traits/remove_cvref.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI

// CRTP base that one can derive from in order to be considered a range adaptor closure
// by the library. When deriving from this class, a pipe operator will be provided to
// make the following hold:
// - `x | f` is equivalent to `f(x)`
// - `f1 | f2` is an adaptor closure `g` such that `g(x)` is equivalent to `f2(f1(x))`
template <class _Tp>
struct __range_adaptor_closure;

// Type that wraps an arbitrary function object and makes it into a range adaptor closure,
// i.e. something that can be called via the `x | f` notation.
template <class _Fn>
struct __range_adaptor_closure_t : _Fn, __range_adaptor_closure<__range_adaptor_closure_t<_Fn>> {
    _LIBCUDACXX_INLINE_VISIBILITY constexpr explicit __range_adaptor_closure_t(_Fn&& __f) : _Fn(_CUDA_VSTD::move(__f)) { }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(__range_adaptor_closure_t);

template <class _Tp>
_LIBCUDACXX_CONCEPT _RangeAdaptorClosure = derived_from<remove_cvref_t<_Tp>, __range_adaptor_closure<remove_cvref_t<_Tp>>>;

template <class _Tp>
struct __range_adaptor_closure {
    _LIBCUDACXX_TEMPLATE(class _View, class _Closure)
      (requires _CUDA_VRANGES::viewable_range<_View> _LIBCUDACXX_AND _RangeAdaptorClosure<_Closure> _LIBCUDACXX_AND
                same_as<_Tp, remove_cvref_t<_Closure>> _LIBCUDACXX_AND
                invocable<_Closure, _View>)
#if !defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE
#endif // !defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr decltype(auto) operator|(_View&& __view, _Closure&& __closure)
        noexcept(is_nothrow_invocable_v<_Closure, _View>)
    { return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Closure>(__closure), _CUDA_VSTD::forward<_View>(__view)); }

    _LIBCUDACXX_TEMPLATE(class _Closure, class _OtherClosure)
      (requires _RangeAdaptorClosure<_Closure> _LIBCUDACXX_AND _RangeAdaptorClosure<_OtherClosure> _LIBCUDACXX_AND
                same_as<_Tp, remove_cvref_t<_Closure>> _LIBCUDACXX_AND
                constructible_from<decay_t<_Closure>, _Closure> _LIBCUDACXX_AND
                constructible_from<decay_t<_OtherClosure>, _OtherClosure>)
#if !defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3)
  _LIBCUDACXX_NODISCARD_ATTRIBUTE
#endif // !defined(_LIBCUDACXX_COMPILER_NVCC_BELOW_11_3)
    _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
    friend constexpr auto operator|(_Closure&& __c1, _OtherClosure&& __c2)
        noexcept(is_nothrow_constructible_v<decay_t<_Closure>, _Closure> &&
                 is_nothrow_constructible_v<decay_t<_OtherClosure>, _OtherClosure>)
    { return __range_adaptor_closure_t(_CUDA_VSTD::__compose(_CUDA_VSTD::forward<_OtherClosure>(__c2), _CUDA_VSTD::forward<_Closure>(__c1))); }
};

_LIBCUDACXX_END_NAMESPACE_RANGES_ABI

#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H
