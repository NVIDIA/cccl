//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_CALL_OR_H
#define _CUDA___FUNCTIONAL_CALL_OR_H

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
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief `__call_or` is an higher-order function that accepts a function, a default
//! value, and arguments to call the function with. If the function is callable with the
//! provided arguments, it invokes the function and returns the result. Otherwise, it
//! returns the default value.
struct __call_or_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Fn, class _Fallback, class... _Args)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<_Fn, _Args...>)
  _CCCL_API constexpr auto operator()(_Fn __fn, _Fallback&&, _Args&&... __args) const
    noexcept(::cuda::std::__is_nothrow_callable_v<_Fn, _Args...>) -> ::cuda::std::__call_result_t<_Fn, _Args...>
  {
    return __fn(static_cast<_Args&&>(__args)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fallback, class... _Args>
  _CCCL_API constexpr auto operator()(::cuda::std::__ignore_t, _Fallback&& __fallback, _Args&&...) const
    noexcept(::cuda::std::is_nothrow_move_constructible_v<_Fallback>) -> _Fallback
  {
    return static_cast<_Fallback&&>(__fallback);
  }
};

_CCCL_GLOBAL_CONSTANT auto __call_or = __call_or_t{};

template <class _Fn, class _Fallback, class... _Args>
using __call_result_or_t _CCCL_NODEBUG_ALIAS = ::cuda::std::__call_result_t<__call_or_t, _Fn, _Fallback, _Args...>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_CALL_OR_H
