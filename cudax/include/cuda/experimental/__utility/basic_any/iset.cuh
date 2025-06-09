//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_ISET_H
#define __CUDAX_DETAIL_BASIC_ANY_ISET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/rtti.cuh>
#include <cuda/experimental/__utility/basic_any/tagged_ptr.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_tables.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//!
//! __iset
//!
template <class... _Interfaces>
struct __iset_
{
  template <class...>
  struct __interface_ : interface<__interface_, extends<_Interfaces...>>
  {};
};

template <class... _Interfaces>
struct __iset : __iset_<_Interfaces...>::template __interface_<>
{};

// flatten any nested sets
template <class _Interface>
using __iset_flatten _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__as_type_list<
  _CUDA_VSTD::
    conditional_t<__is_specialization_of_v<_Interface, __iset>, _Interface, _CUDA_VSTD::__type_list<_Interface>>>;

// flatten all sets into one, remove duplicates, and sort the elements.
// TODO: sort!
// template <class... _Interfaces>
// using iset _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<
//   _CUDA_VSTD::__type_unique<_CUDA_VSTD::__type_sort<_CUDA_VSTD::__type_concat<__iset_flatten<_Interfaces>...>>>,
//   _CUDA_VSTD::__type_quote<__iset>>;
template <class... _Interfaces>
using iset =
  _CUDA_VSTD::__type_call<_CUDA_VSTD::__type_unique<_CUDA_VSTD::__type_concat<__iset_flatten<_Interfaces>...>>,
                          _CUDA_VSTD::__type_quote<__iset>>;

//!
//! Virtual table pointers
//!
template <class... _Interfaces>
struct __iset_vptr : __base_vptr
{
  using __iset_vtable _CCCL_NODEBUG_ALIAS = __vtable_for<__iset<_Interfaces...>>;

  __iset_vptr() = default;

  _CCCL_HOST_API constexpr __iset_vptr(__iset_vtable const* __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  _CCCL_HOST_API constexpr __iset_vptr(__base_vptr __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  // Permit narrowing conversions from a super-set __vptr. Warning: we can't
  // simply constrain this because then the ctor from __base_vptr would be
  // selected instead, giving the wrong result.
  template <class... _Others>
  _CCCL_HOST_API __iset_vptr(__iset_vptr<_Others...> __vptr) noexcept
      : __base_vptr(__vptr->__query_interface(iunknown()))
  {
    static_assert(_CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Others...>, _Interfaces...>, "");
    _CCCL_ASSERT(__vptr_->__kind_ == __vtable_kind::__rtti && __vptr_->__cookie_ == 0xDEADBEEF,
                 "query_interface returned a bad pointer to the iunknown vtable");
  }

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API constexpr auto operator->() const noexcept -> __iset_vptr const*
  {
    return this;
  }

  template <class _Interface>
  [[nodiscard]] _CCCL_TRIVIAL_HOST_API constexpr auto __query_interface(_Interface) const noexcept
    -> __vptr_for<_Interface>
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
  _CCCL_TRIVIAL_HOST_API auto __set(__iset_vptr<_Interfaces...> __vptr, bool __flag) noexcept -> void
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__vptr.__vptr_) | uintptr_t(__flag);
  }

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto __get() const noexcept -> __iset_vptr<_Interfaces...>
  {
    return __iset_vptr<_Interfaces...>{reinterpret_cast<__rtti_base const*>(__ptr_ & ~uintptr_t(1))};
  }

  [[nodiscard]] _CCCL_TRIVIAL_HOST_API auto __flag() const noexcept -> bool
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_BASIC_ANY_ISET_H
