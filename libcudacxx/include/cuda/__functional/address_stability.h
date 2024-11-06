//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H
#define _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/move.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! Trait telling whether a function object type F does not rely on the memory addresses of its arguments. The nested
//! value is true when the addresses of the arguments do not matter and arguments can be provided from arbitrary copies
//! of the respective sources. This trait can be specialized for custom function objects types.
//! @see proclaim_copyable_arguments
template <typename F, typename SFINAE = void>
struct proclaims_copyable_arguments : _CUDA_VSTD::false_type
{};

#if !defined(_CCCL_NO_VARIABLE_TEMPLATES)
template <typename F, typename... Args>
_CCCL_INLINE_VAR constexpr bool proclaims_copyable_arguments_v = proclaims_copyable_arguments<F, Args...>::value;
#endif // !_CCCL_NO_VARIABLE_TEMPLATES

// Wrapper for a callable to mark it as permitting copied arguments
template <typename F>
struct __callable_permitting_copied_arguments : F
{
  using F::operator();
};

template <typename F>
struct proclaims_copyable_arguments<__callable_permitting_copied_arguments<F>> : _CUDA_VSTD::true_type
{};

//! Creates a new function object from an existing one, which is marked as permitting its arguments to be copies of
//! whatever source they come from. This implies that the addresses of the arguments are irrelevant to the function
//! object.
//! @see proclaims_copyable_arguments
template <typename F>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
proclaim_copyable_arguments(F f) -> __callable_permitting_copied_arguments<F>
{
  return __callable_permitting_copied_arguments<F>{_CUDA_VSTD::move(f)};
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___FUNCTIONAL_ADDRESS_STABILITY_H
