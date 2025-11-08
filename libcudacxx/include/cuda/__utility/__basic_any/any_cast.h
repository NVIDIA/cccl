//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_ANY_CAST_H
#define _CUDA___UTILITY_BASIC_ANY_ANY_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/access.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! __valid_any_cast
//!
template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast = true;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp> = false;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp*> =
  !::cuda::std::is_const_v<_Interface> || ::cuda::std::is_const_v<_Tp>;

//!
//! __any_cast
//!
_CCCL_TEMPLATE(class _Tp, class _Interface)
_CCCL_REQUIRES(__satisfies<_Tp, _Interface> || ::cuda::std::is_void_v<_Tp>)
[[nodiscard]] _CCCL_API auto __any_cast(__basic_any<_Interface>* __self) noexcept -> _Tp*
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (::cuda::std::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

_CCCL_TEMPLATE(class _Tp, class _Interface)
_CCCL_REQUIRES(__satisfies<_Tp, _Interface> || ::cuda::std::is_void_v<_Tp>)
[[nodiscard]] _CCCL_API auto __any_cast(__basic_any<_Interface> const* __self) noexcept -> _Tp const*
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (::cuda::std::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp const*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

// TODO: implement the same overloads as for std::__any_cast

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_ANY_CAST_H
