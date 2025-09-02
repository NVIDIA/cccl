//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_OVERRIDES_H
#define _CUDA___UTILITY_BASIC_ANY_OVERRIDES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_const.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Interface, class _Tp = __remove_ireference_t<_Interface>>
using __overrides_for_t _CCCL_NODEBUG_ALIAS = typename _Interface::template overrides<_Tp>;

//!
//! __overrides_for
//!
template <class _InterfaceOrModel, class... _VirtualFnsOrOverrides>
struct __overrides_list
{
  static_assert(!::cuda::std::is_const_v<_InterfaceOrModel>, "expected a class type");
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<_InterfaceOrModel, _VirtualFnsOrOverrides::value...>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __vtable const*;
};

template <class... _Interfaces>
struct __overrides_list<__iset_<_Interfaces...>>
{
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<__iset_<_Interfaces...>>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __iset_vptr<_Interfaces...>;
};

template <>
struct __overrides_list<__iunknown>
{
  using __vtable _CCCL_NODEBUG_ALIAS = ::cuda::std::__ignore_t; // no vtable, rtti is added explicitly in __vtable_tuple
  using __vptr_t _CCCL_NODEBUG_ALIAS = __rtti const*;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_OVERRIDES_H
