// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_ALLOCATOR_ARG_T_H
#define _CUDA_STD___FUNCTIONAL_ALLOCATOR_ARG_T_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/uses_allocator.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT allocator_arg_t
{
  _CCCL_HIDE_FROM_ABI explicit allocator_arg_t() = default;
};

inline constexpr allocator_arg_t allocator_arg = allocator_arg_t();

// allocator construction

//! @brief Which uses-allocator constructor to call.
enum class __select_uses_allocator
{
  __none, //!< Construct without the allocator.
  __allocator_arg, //!< Construct from `allocator_arg`, the allocator, then the arguments.
  __allocator_last, //!< Construct from the arguments, then the allocator.
};

using __select_uses_allocator_none = integral_constant<__select_uses_allocator, __select_uses_allocator::__none>;
using __select_uses_allocator_arg =
  integral_constant<__select_uses_allocator, __select_uses_allocator::__allocator_arg>;
using __select_uses_allocator_last =
  integral_constant<__select_uses_allocator, __select_uses_allocator::__allocator_last>;

template <class _Tp, class _Alloc, class... _Args>
inline constexpr __select_uses_allocator __select_uses_allocator_v =
  !uses_allocator<_Tp, remove_cvref_t<_Alloc>>::value ? __select_uses_allocator::__none
  : is_constructible_v<_Tp, allocator_arg_t, _Alloc, _Args...>
    ? __select_uses_allocator::__allocator_arg
    : __select_uses_allocator::__allocator_last;

template <class _Tp, class _Alloc, class... _Args>
struct __uses_alloc_ctor : integral_constant<__select_uses_allocator, __select_uses_allocator_v<_Tp, _Alloc, _Args...>>
{};

template <__select_uses_allocator _Kind, class _Tp, class _Alloc, class... _Args>
inline constexpr bool __is_nothrow_uses_allocator_constructible = false;

template <class _Tp, class _Alloc, class... _Args>
inline constexpr bool __is_nothrow_uses_allocator_constructible<__select_uses_allocator::__none, _Tp, _Alloc, _Args...> =
  is_nothrow_constructible_v<_Tp, _Args...>;

template <class _Tp, class _Alloc, class... _Args>
inline constexpr bool
  __is_nothrow_uses_allocator_constructible<__select_uses_allocator::__allocator_arg, _Tp, _Alloc, _Args...> =
    is_nothrow_constructible_v<_Tp, allocator_arg_t, const _Alloc&, _Args...>;

template <class _Tp, class _Alloc, class... _Args>
inline constexpr bool
  __is_nothrow_uses_allocator_constructible<__select_uses_allocator::__allocator_last, _Tp, _Alloc, _Args...> =
    is_nothrow_constructible_v<_Tp, _Args..., const _Alloc&>;

template <class _Tp, class _Alloc, class... _Args>
inline constexpr bool __is_nothrow_uses_allocator_constructible_v =
  __is_nothrow_uses_allocator_constructible<__select_uses_allocator_v<_Tp, _Alloc, _Args...>, _Tp, _Alloc, _Args...>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_ALLOCATOR_ARG_T_H
