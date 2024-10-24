// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H
#define _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/invocable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__functional/compose.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017 && !defined(_CCCL_COMPILER_MSVC_2017)

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
struct __range_adaptor_closure_t
    : _Fn
    , __range_adaptor_closure<__range_adaptor_closure_t<_Fn>>
{
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __range_adaptor_closure_t(_Fn&& __f)
      : _Fn(_CUDA_VSTD::move(__f))
  {}
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(__range_adaptor_closure_t);

template <class _Tp>
_LIBCUDACXX_CONCEPT _RangeAdaptorClosure =
  derived_from<remove_cvref_t<_Tp>, __range_adaptor_closure<remove_cvref_t<_Tp>>>;

template <class _Tp, class _View, class _Closure>
_LIBCUDACXX_CONCEPT __range_adaptor_can_pipe_invoke =
  same_as<_Tp, remove_cvref_t<_Closure>> && _CUDA_VRANGES::viewable_range<_View> && _RangeAdaptorClosure<_Closure>
  && invocable<_Closure, _View>;

template <class _Tp, class _Closure, class _OtherClosure>
_LIBCUDACXX_CONCEPT __range_adaptor_can_pipe_compose =
  _RangeAdaptorClosure<_Closure> && _RangeAdaptorClosure<_OtherClosure> && same_as<_Tp, remove_cvref_t<_Closure>>
  && constructible_from<decay_t<_Closure>, _Closure> && constructible_from<decay_t<_OtherClosure>, _OtherClosure>;

template <class _Tp>
struct __range_adaptor_closure
{
#  if defined(_CCCL_COMPILER_NVRTC)
  template <class _View, class _Closure>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
  operator|(_View&& __view, _Closure&& __closure) noexcept(is_nothrow_invocable_v<_Closure, _View>)
    _LIBCUDACXX_TRAILING_REQUIRES(invoke_result_t<_Closure, _View>)(
      __range_adaptor_can_pipe_invoke<_Tp, _View, _Closure>)
  {
    return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Closure>(__closure), _CUDA_VSTD::forward<_View>(__view));
  }
#  else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv !_CCCL_COMPILER_NVRTC vvv
  _LIBCUDACXX_TEMPLATE(class _View, class _Closure)
  _LIBCUDACXX_REQUIRES(__range_adaptor_can_pipe_invoke<_Tp, _View, _Closure>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr decltype(auto)
  operator|(_View&& __view, _Closure&& __closure) noexcept(is_nothrow_invocable_v<_Closure, _View>)
  {
    return _CUDA_VSTD::invoke(_CUDA_VSTD::forward<_Closure>(__closure), _CUDA_VSTD::forward<_View>(__view));
  }
#  endif // !_CCCL_COMPILER_NVRTC

#  if defined(_CCCL_COMPILER_NVRTC)
  template <class _Closure, class _OtherClosure>
  using __compose_return_t = decltype(__range_adaptor_closure_t(
    _CUDA_VSTD::__compose(_CUDA_VSTD::declval<_OtherClosure>(), _CUDA_VSTD::declval<_Closure>())));

  template <class _Closure, class _OtherClosure>
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator|(_Closure&& __c1, _OtherClosure&& __c2) //
    noexcept(is_nothrow_constructible_v<decay_t<_Closure>, _Closure>
             && is_nothrow_constructible_v<decay_t<_OtherClosure>, _OtherClosure>)
      _LIBCUDACXX_TRAILING_REQUIRES(__compose_return_t<_Closure, _OtherClosure>)(
        __range_adaptor_can_pipe_compose<_Tp, _Closure, _OtherClosure>)
  {
    return __range_adaptor_closure_t(
      _CUDA_VSTD::__compose(_CUDA_VSTD::forward<_OtherClosure>(__c2), _CUDA_VSTD::forward<_Closure>(__c1)));
  }
#  else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv !_CCCL_COMPILER_NVRTC vvv
  _LIBCUDACXX_TEMPLATE(class _Closure, class _OtherClosure)
  _LIBCUDACXX_REQUIRES(__range_adaptor_can_pipe_compose<_Tp, _Closure, _OtherClosure>)
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator|(_Closure&& __c1, _OtherClosure&& __c2) //
    noexcept(is_nothrow_constructible_v<decay_t<_Closure>, _Closure>
             && is_nothrow_constructible_v<decay_t<_OtherClosure>, _OtherClosure>)
  {
    return __range_adaptor_closure_t(
      _CUDA_VSTD::__compose(_CUDA_VSTD::forward<_OtherClosure>(__c2), _CUDA_VSTD::forward<_Closure>(__c1)));
  }
#  endif // !_CCCL_COMPILER_NVRTC
};

#endif // _CCCL_STD_VER >= 2017 && !_CCCL_COMPILER_MSVC_2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___RANGES_RANGE_ADAPTOR_H
