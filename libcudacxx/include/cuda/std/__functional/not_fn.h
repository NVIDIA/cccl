// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_NOT_FN_H
#define _CUDA_STD___FUNCTIONAL_NOT_FN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Fn, class... _Args>
_CCCL_CONCEPT __can_invoke_and_negate = _CCCL_REQUIRES_EXPR((_Fn, variadic _Args), _Fn&& __f, _Args&&... __args)(
  (!::cuda::std::invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)...)));

template <class _Fn>
struct __not_fn_t
{
  _Fn __f;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(is_constructible_v<_Fn, _Args&&...>)
  _CCCL_API explicit constexpr __not_fn_t(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Fn, _Args&&...>)
      : __f(::cuda::std::forward<_Args>(__args)...)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__can_invoke_and_negate<_Fn&, _Args...>)
  _CCCL_API constexpr auto
  operator()(_Args&&... __args) & noexcept(noexcept(!::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(!::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...))
  {
    return !::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!__can_invoke_and_negate<_Fn&, _Args...>) )
  void operator()(_Args&&...) & = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__can_invoke_and_negate<const _Fn&, _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) const& noexcept(
    noexcept(!::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(!::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...))
  {
    return !::cuda::std::invoke(__f, ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!__can_invoke_and_negate<const _Fn&, _Args...>) )
  void operator()(_Args&&...) const& = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__can_invoke_and_negate<_Fn, _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) && noexcept(
    noexcept(!::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(!::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...))
  {
    return !::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!__can_invoke_and_negate<_Fn, _Args...>) )
  void operator()(_Args&&...) && = delete;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__can_invoke_and_negate<const _Fn, _Args...>)
  _CCCL_API constexpr auto operator()(_Args&&... __args) const&& noexcept(
    noexcept(!::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...)))
    -> decltype(!::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...))
  {
    return !::cuda::std::invoke(::cuda::std::move(__f), ::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!__can_invoke_and_negate<const _Fn, _Args...>) )
  void operator()(_Args&&...) const&& = delete;
};

_CCCL_TEMPLATE(class _Fn)
_CCCL_REQUIRES(is_constructible_v<decay_t<_Fn>, _Fn> _CCCL_AND is_move_constructible_v<decay_t<_Fn>>)
[[nodiscard]] _CCCL_API constexpr auto not_fn(_Fn&& __f)
{
  return __not_fn_t<decay_t<_Fn>>(::cuda::std::forward<_Fn>(__f));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_NOT_FN_H
