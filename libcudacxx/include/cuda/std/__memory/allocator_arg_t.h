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

#ifndef _LIBCUDACXX___FUNCTIONAL_ALLOCATOR_ARG_T_H
#define _LIBCUDACXX___FUNCTIONAL_ALLOCATOR_ARG_T_H

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
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _CCCL_TYPE_VISIBILITY_DEFAULT allocator_arg_t
{
  _CCCL_HIDE_FROM_ABI explicit allocator_arg_t() = default;
};

_LIBCUDACXX_INLINE_VAR constexpr allocator_arg_t allocator_arg = allocator_arg_t();

// allocator construction

template <class _Tp, class _Alloc, class... _Args>
struct __uses_alloc_ctor_imp
{
  typedef _LIBCUDACXX_NODEBUG_TYPE __remove_cvref_t<_Alloc> _RawAlloc;
  static const bool __ua = uses_allocator<_Tp, _RawAlloc>::value;
  static const bool __ic = is_constructible<_Tp, allocator_arg_t, _Alloc, _Args...>::value;
  static const int value = __ua ? 2 - __ic : 0;
};

template <class _Tp, class _Alloc, class... _Args>
struct __uses_alloc_ctor : integral_constant<int, __uses_alloc_ctor_imp<_Tp, _Alloc, _Args...>::value>
{};

template <class _Tp, class _Allocator, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI void
__user_alloc_construct_impl(integral_constant<int, 0>, _Tp* __storage, const _Allocator&, _Args&&... __args)
{
  new (__storage) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
}

// FIXME: This should have a version which takes a non-const alloc.
template <class _Tp, class _Allocator, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI void
__user_alloc_construct_impl(integral_constant<int, 1>, _Tp* __storage, const _Allocator& __a, _Args&&... __args)
{
  new (__storage) _Tp(allocator_arg, __a, _CUDA_VSTD::forward<_Args>(__args)...);
}

// FIXME: This should have a version which takes a non-const alloc.
template <class _Tp, class _Allocator, class... _Args>
_LIBCUDACXX_HIDE_FROM_ABI void
__user_alloc_construct_impl(integral_constant<int, 2>, _Tp* __storage, const _Allocator& __a, _Args&&... __args)
{
  new (__storage) _Tp(_CUDA_VSTD::forward<_Args>(__args)..., __a);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_ALLOCATOR_ARG_T_H
