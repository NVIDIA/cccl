//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_NO_UNIQUE_MEMBER_H
#define _CUDA_STD___UTILITY_NO_UNIQUE_MEMBER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief A replacement for [[no_unique_address]] members. The class should be used as a base class. The stored class
//! can be accessed by the __get() method. If there is a possiiblity that 2 or more members could have the same type,
//! each __no_unique_member base class should be supplied with different id template parameter.
template <class _Tp, size_t _Id = 0, bool = is_empty_v<_Tp>>
class __no_unique_member
{
  _Tp __value_;

public:
  _CCCL_HIDE_FROM_ABI constexpr __no_unique_member() noexcept(is_nothrow_default_constructible_v<_Tp>) = default;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) > 0) _CCCL_AND is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr __no_unique_member(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : __value_(::cuda::std::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return __value_;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return __value_;
  }
};

template <class _Tp, size_t _Id>
class _CCCL_DECLSPEC_EMPTY_BASES __no_unique_member<_Tp, _Id, true> : _Tp
{
public:
  _CCCL_HIDE_FROM_ABI constexpr __no_unique_member() noexcept(is_nothrow_default_constructible_v<_Tp>) = default;

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((sizeof...(_Args) > 0) _CCCL_AND is_constructible_v<_Tp, _Args...>)
  _CCCL_API constexpr __no_unique_member(_Args&&... __args) noexcept(is_nothrow_constructible_v<_Tp, _Args...>)
      : _Tp(::cuda::std::forward<_Args>(__args)...)
  {}

  [[nodiscard]] _CCCL_API constexpr _Tp& __get() noexcept
  {
    return *this;
  }

  [[nodiscard]] _CCCL_API constexpr const _Tp& __get() const noexcept
  {
    return *this;
  }
};

//! @brief Gets the specified no unique member of an object. Should be used as:
//!
//! struct S : __no_unique_member<int>
//! {
//!   using _Member = __no_unique_member<int>;
//!
//!   // ...
//! };
//!
//! void fn(S& s)
//! {
//!   _CCCL_GET_NO_UNIQUE_MEMBER(s, _Member) = 10;
//! }
#define _CCCL_GET_NO_UNIQUE_MEMBER(_OBJ, _MEMBER)                                                             \
  static_cast<::cuda::std::__copy_cv_t<::cuda::std::remove_reference_t<decltype(_OBJ)>,                       \
                                       typename ::cuda::std::remove_cvref_t<decltype(_OBJ)>::_MEMBER>&>(_OBJ) \
    .__get()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_NO_UNIQUE_MEMBER_H
