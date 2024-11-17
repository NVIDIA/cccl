//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_ANY_CAST_H
#define __CUDAX_DETAIL_BASIC_ANY_ANY_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>

namespace cuda::experimental
{
///
/// __valid_any_cast
///
template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast = true;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp> = false;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp*> =
  !_CUDA_VSTD::is_const_v<_Interface> || _CUDA_VSTD::is_const_v<_Tp>;

///
/// any_cast
///
_LIBCUDACXX_TEMPLATE(class _Tp, class _Interface)
_LIBCUDACXX_REQUIRES(__satisfies<_Tp, _Interface> || _CUDA_VSTD::is_void_v<_Tp>)
_CCCL_NODISCARD _CUDAX_API _Tp* any_cast(basic_any<_Interface>* __self) noexcept
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (_CUDA_VSTD::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Interface)
_LIBCUDACXX_REQUIRES(__satisfies<_Tp, _Interface> || _CUDA_VSTD::is_void_v<_Tp>)
_CCCL_NODISCARD _CUDAX_API _Tp const* any_cast(basic_any<_Interface> const* __self) noexcept
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (_CUDA_VSTD::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp const*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

// TODO: implement the same overloads as for std::any_cast

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_ANY_CAST_H
