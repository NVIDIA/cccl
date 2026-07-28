//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_LAZY_CALL_OR_H
#define _CUDA___FUNCTIONAL_LAZY_CALL_OR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CPO(__lazy_call_or_ns)
//! @brief `__lazy_call_or` is like `__call_or` except that the fallback value is computed
//! lazily.
//!
//! The fallback value must be a functor that takes no arguments and returns a single
//! value. The type of the returned fallback value need not be the same as the type of the
//! computed value.
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn, class _FallbackCallable, class... _Args)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<_Fn, _Args...>)
  _CCCL_API constexpr auto operator()(_Fn __fn, _FallbackCallable&&, _Args&&... __args) const
    noexcept(::cuda::std::__is_nothrow_callable_v<_Fn, _Args...>) -> ::cuda::std::__call_result_t<_Fn, _Args...>
  {
    return __fn(::cuda::std::forward<_Args>(__args)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _FallbackCallable, class... _Args>
  _CCCL_API constexpr auto operator()(::cuda::std::__ignore_t, _FallbackCallable&& __fallback, _Args&&...) const
    noexcept(::cuda::std::__is_nothrow_callable_v<_FallbackCallable>) -> ::cuda::std::__call_result_t<_FallbackCallable>
  {
    return ::cuda::std::forward<_FallbackCallable>(__fallback)();
  }
};
_CCCL_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto __lazy_call_or = __lazy_call_or_ns::__fn{};
} // namespace __cpo

template <class _Fn, class _FallbackCallable, class... _Args>
using __lazy_call_result_or_t _CCCL_NODEBUG_ALIAS =
  ::cuda::std::__call_result_t<__lazy_call_or_ns::__fn, _Fn, _FallbackCallable, _Args...>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_LAZY_CALL_OR_H
