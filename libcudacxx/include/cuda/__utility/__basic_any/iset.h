//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_ISET_H
#define _CUDA___UTILITY_BASIC_ANY_ISET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/rtti.h>
#include <cuda/__utility/__basic_any/tagged_ptr.h>
#include <cuda/__utility/__basic_any/virtual_tables.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! __iset_
//!
template <class... _Interfaces>
struct __iset__
{
  template <class...>
  struct __interface_ : __basic_interface<__interface_, __extends<_Interfaces...>>
  {};
};

template <class... _Interfaces>
struct __iset_ : __iset__<_Interfaces...>::template __interface_<>
{};

// flatten any nested sets
template <class _Interface>
using __iset_flatten _CCCL_NODEBUG_ALIAS = ::cuda::std::__as_type_list<
  ::cuda::std::
    conditional_t<__is_specialization_of_v<_Interface, __iset_>, _Interface, ::cuda::std::__type_list<_Interface>>>;

// flatten all sets into one, remove duplicates, and sort the elements.
// TODO: sort!
// template <class... _Interfaces>
// using __iset _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_call<
//   ::cuda::std::__type_unique<::cuda::std::__type_sort<::cuda::std::__type_concat<__iset_flatten<_Interfaces>...>>>,
//   ::cuda::std::__type_quote<__iset_>>;
template <class... _Interfaces>
using __iset =
  ::cuda::std::__type_call<::cuda::std::__type_unique<::cuda::std::__type_concat<__iset_flatten<_Interfaces>...>>,
                           ::cuda::std::__type_quote<__iset_>>;

//!
//! Virtual table pointers
//!
template <class... _Interfaces>
struct __iset_vptr : __base_vptr
{
  using __iset_vtable _CCCL_NODEBUG_ALIAS = __vtable_for<__iset_<_Interfaces...>>;

  __iset_vptr() = default;

  _CCCL_API constexpr __iset_vptr(__iset_vtable const* __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  _CCCL_API constexpr __iset_vptr(__base_vptr __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  // Permit narrowing conversions from a super-set __vptr. Warning: we can't
  // simply constrain this because then the ctor from __base_vptr would be
  // selected instead, giving the wrong result.
  template <class... _Others>
  _CCCL_API __iset_vptr(__iset_vptr<_Others...> __vptr) noexcept
      : __base_vptr(__vptr->__query_interface(__iunknown()))
  {
    static_assert(::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<_Others...>, _Interfaces...>, "");
    _CCCL_ASSERT(__vptr_->__kind_ == __vtable_kind::__rtti && __vptr_->__cookie_ == 0xDEADBEEF,
                 "query_interface returned a bad pointer to the __iunknown vtable");
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator->() const noexcept -> __iset_vptr const*
  {
    return this;
  }

  template <class _Interface>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto __query_interface(_Interface) const noexcept -> __vptr_for<_Interface>
  {
    if (__vptr_->__kind_ == __vtable_kind::__normal)
    {
      return static_cast<__iset_vtable const*>(__vptr_)->__query_interface(_Interface{});
    }
    else
    {
      return static_cast<__rtti const*>(__vptr_)->__query_interface(_Interface{});
    }
  }
};

template <class... _Interfaces>
struct __tagged_ptr<__iset_vptr<_Interfaces...>>
{
  _CCCL_NODEBUG_API auto __set(__iset_vptr<_Interfaces...> __vptr, bool __flag) noexcept -> void
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__vptr.__vptr_) | uintptr_t(__flag);
  }

  [[nodiscard]] _CCCL_NODEBUG_API auto __get() const noexcept -> __iset_vptr<_Interfaces...>
  {
    return __iset_vptr<_Interfaces...>{reinterpret_cast<__rtti_base const*>(__ptr_ & ~uintptr_t(1))};
  }

  [[nodiscard]] _CCCL_NODEBUG_API auto __flag() const noexcept -> bool
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_ISET_H
