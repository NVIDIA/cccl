//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ADDRESS_STABILITY_H
#define _LIBCUDACXX___TYPE_TRAITS_ADDRESS_STABILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// need a separate implementation trait because we SFINAE with a type parameter before the variadic pack
template <typename F, typename SFINAE, typename... Args>
struct __allows_copied_arguments_impl : _CUDA_VSTD::false_type
{};

template <typename F, typename... Args>
struct __allows_copied_arguments_impl<F, _CUDA_VSTD::void_t<decltype(F::allows_copied_arguments)>, Args...>
{
  static constexpr bool value = F::allows_copied_arguments;
};

//! Trait telling whether a function object relies on the memory address of its arguments when called with the given set
//! of types. The nested value is true when the addresses of the arguments do not matter and arguments can be provided
//! from arbitrary copies of the respective sources. Can be specialized for custom function objects and parameter types.
template <typename F, typename... Args>
struct allows_copied_arguments : __allows_copied_arguments_impl<F, void, Args...>
{};

#if _CCCL_STD_VER >= 2014
template <typename F, typename... Args>
_LIBCUDACXX_INLINE_VAR constexpr bool allows_copied_arguments_v = allows_copied_arguments<F, Args...>::value;
#endif // _CCCL_STD_VER >= 2014

//! Wrapper for a callable to mark it as allowing copied arguments
template <typename F>
struct callable_allowing_copied_arguments : F
{
  using F::operator();
  static constexpr bool allows_copied_arguments = true;
};

//! Creates a new function object from an existing one, allowing its arguments to be copies of whatever source they come
//! from. This implies that the addresses of the arguments are irrelevant to the function object.
template <typename F>
_CCCL_HOST_DEVICE constexpr auto allow_copied_arguments(F f) -> callable_allowing_copied_arguments<F>
{
  return callable_allowing_copied_arguments<F>{_CUDA_VSTD::move(f)};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___TYPE_TRAITS_ADDRESS_STABILITY_H
