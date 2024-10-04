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

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/negation.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _FuncPtr>
struct __all_parameters_by_value_fptr : _CUDA_VSTD::false_type
{};

// TODO(bgruber): does the reference detection even work for proxy references? If a callable takes a
// thrust::device_reference<T> or a std::reference_wrapper<T>, allowing a copy would be wrong

template <typename R, typename... Args>
struct __all_parameters_by_value_fptr<R (*)(Args...)>
    : _CUDA_VSTD::conjunction<_CUDA_VSTD::_Not<_CUDA_VSTD::is_reference<Args>>...>
{};

template <typename R, typename C, typename... Args>
struct __all_parameters_by_value_fptr<R (C::*)(Args...)>
    : _CUDA_VSTD::conjunction<_CUDA_VSTD::_Not<_CUDA_VSTD::is_reference<Args>>...>
{};

template <typename R, typename C, typename... Args>
struct __all_parameters_by_value_fptr<R (C::*)(Args...) const>
    : _CUDA_VSTD::conjunction<_CUDA_VSTD::_Not<_CUDA_VSTD::is_reference<Args>>...>
{};

// case for when we cannot address the call target
template <typename _F, typename _SFINAE = void>
struct __all_parameters_by_value : _CUDA_VSTD::false_type
{};

// case for function pointers
template <typename _FP>
struct __all_parameters_by_value<
  _FP,
  _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_pointer<_FP>::value
                            && _CUDA_VSTD::is_function<_CUDA_VSTD::__remove_pointer_t<_FP>>::value>>
    : __all_parameters_by_value_fptr<_FP>
{};

// case for function objects
template <typename _F>
struct __all_parameters_by_value<_F, _CUDA_VSTD::void_t<decltype(&_F::operator())>>
    : __all_parameters_by_value_fptr<decltype(&_F::operator())>
{};

// case for functions
template <typename _F>
struct __all_parameters_by_value<_F, _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_function<_F>::value>>
    : __all_parameters_by_value_fptr<decltype(&::cuda::std::declval<_F>())>
{};

template <typename _T>
using __value_type = _CUDA_VSTD::bool_constant<!_CUDA_VSTD::is_class<_T>::value && !_CUDA_VSTD::is_union<_T>::value
                                               && !_CUDA_VSTD::is_reference<_T>::value>;

// need a separate implementation trait because we SFINAE with a type parameter before the variadic pack
// TODO(bgruber): if we ever get something like declcall (https://wg21.link/P2825), we should use it here to inspect the
// signature of the function that overload resolution chose
template <typename _F, typename _SFINAE, typename... _Args>
struct __allows_copied_arguments_impl : _CUDA_VSTD::conjunction<__all_parameters_by_value<_F>, __value_type<_Args>...>
{};

template <typename F, typename... Args>
struct __allows_copied_arguments_impl<F, _CUDA_VSTD::void_t<decltype(F::allows_copied_arguments)>, Args...>
{
  static constexpr bool value = F::allows_copied_arguments;
};

//! Trait telling whether a function object type, function type, or function pointer type relies on the memory address
//! of its arguments when called with the given set of types. The nested value is true when the addresses of the
//! arguments do not matter and arguments can be provided from arbitrary copies of the respective sources. Can be
//! specialized for custom function objects and parameter types.
template <typename F, typename... Args>
struct allows_copied_arguments : __allows_copied_arguments_impl<F, void, Args...>
{};

#if _CCCL_STD_VER >= 2014
template <typename F, typename... Args>
_LIBCUDACXX_INLINE_VAR constexpr bool allows_copied_arguments_v = allows_copied_arguments<F, Args...>::value;
#endif // _CCCL_STD_VER >= 2014

//! Wrapper for a callable to mark it as allowing copied arguments
template <typename _F, typename _SFINAE = void>
struct callable_allowing_copied_arguments : _F
{
  using _F::operator();
  static constexpr bool allows_copied_arguments = true;
};

// TODO(bgruber): maybe just provide one implementation that stores the callable as a member
template <typename _FP>
struct callable_allowing_copied_arguments<
  _FP,
  _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_pointer<_FP>::value
                            && _CUDA_VSTD::is_function<_CUDA_VSTD::__remove_pointer_t<_FP>>::value>>
{
  _FP __fp;

  // TODO(bgruber): we may just use ::cuda::std::invoke() here
  template <typename... _Args>
  auto operator()(_Args&&... args) const -> decltype(__fp(_CUDA_VSTD::forward<_Args>(args)...))
  {
    return __fp(_CUDA_VSTD::forward<_Args>(args)...);
  }

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
