// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___RANGES_RANGE_ADAPTOR_H
#define _CUDA_STD___RANGES_RANGE_ADAPTOR_H

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

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_RANGES

// CRTP base that one can derive from in order to be considered a range adaptor closure
// by the library. When deriving from this class, a pipe operator will be provided to
// make the following hold:
// - `x | f` is equivalent to `f(x)`
// - `f1 | f2` is an adaptor closure `g` such that `g(x)` is equivalent to `f2(f1(x))`
template <class _Tp, enable_if_t<is_class_v<_Tp>, int> = 0, enable_if_t<same_as<_Tp, remove_cv_t<_Tp>>, int> = 0>
struct __range_adaptor_closure
{};

// Type that wraps an arbitrary function object and makes it into a range adaptor closure,
// i.e. something that can be called via the `x | f` notation.
template <class _Fn>
struct __pipeable
    : _Fn
    , __range_adaptor_closure<__pipeable<_Fn>>
{
  _CCCL_API constexpr explicit __pipeable(_Fn&& __f)
      : _Fn(::cuda::std::move(__f))
  {}
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(__pipeable);

template <class _Tp>
_CCCL_HOST_DEVICE _Tp __derived_from_range_adaptor_closure(__range_adaptor_closure<_Tp>*);

template <class _Tp>
_CCCL_CONCEPT __is_range_adaptor_closure = _CCCL_REQUIRES_EXPR(
  (_Tp))(requires(!::cuda::std::ranges::range<remove_cvref_t<_Tp>>),
         _Same_as(remove_cvref_t<_Tp>)::cuda::std::ranges::__derived_from_range_adaptor_closure(
           (remove_cvref_t<_Tp>*) nullptr));

template <class _Range, class _Closure>
_CCCL_CONCEPT __range_adaptor_can_pipe_invoke = _CCCL_REQUIRES_EXPR((_Range, _Closure))(
  requires(range<_Range>), requires(__is_range_adaptor_closure<_Closure>), requires(invocable<_Closure, _Range>));

template <class _Closure, class _OtherClosure>
_CCCL_CONCEPT __range_adaptor_can_pipe_compose = _CCCL_REQUIRES_EXPR((_Closure, _OtherClosure))(
  requires(__is_range_adaptor_closure<_Closure>),
  requires(__is_range_adaptor_closure<_OtherClosure>),
  requires(constructible_from<decay_t<_Closure>, _Closure>),
  requires(constructible_from<decay_t<_OtherClosure>, _OtherClosure>));

_CCCL_TEMPLATE(class _Range, class _Closure)
_CCCL_REQUIRES(__range_adaptor_can_pipe_invoke<_Range, _Closure>)
[[nodiscard]] _CCCL_API constexpr decltype(auto)
operator|(_Range&& __range, _Closure&& __closure) noexcept(is_nothrow_invocable_v<_Closure, _Range>)
{
  return ::cuda::std::invoke(::cuda::std::forward<_Closure>(__closure), ::cuda::std::forward<_Range>(__range));
}

_CCCL_TEMPLATE(class _Closure, class _OtherClosure)
_CCCL_REQUIRES(__range_adaptor_can_pipe_compose<_Closure, _OtherClosure>)
[[nodiscard]] _CCCL_API constexpr auto operator|(_Closure&& __closure, _OtherClosure&& __other_closure) noexcept(
  is_nothrow_constructible_v<decay_t<_Closure>, _Closure>
  && is_nothrow_constructible_v<decay_t<_OtherClosure>, _OtherClosure>)
{
  return __pipeable(::cuda::std::__compose(
    ::cuda::std::forward<_OtherClosure>(__other_closure), ::cuda::std::forward<_Closure>(__closure)));
}

template <class _Tp, enable_if_t<is_class_v<_Tp>, int> = 0, enable_if_t<same_as<_Tp, remove_cv_t<_Tp>>, int> = 0>
class range_adaptor_closure : public __range_adaptor_closure<_Tp>
{};

_CCCL_END_NAMESPACE_RANGES

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___RANGES_RANGE_ADAPTOR_H
