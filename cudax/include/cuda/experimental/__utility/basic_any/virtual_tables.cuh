//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_VIRTUAL_TABLES_H
#define __CUDAX_DETAIL_BASIC_ANY_VIRTUAL_TABLES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/rtti.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_functions.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_ptrs.cuh>

namespace cuda::experimental
{
template <class _Interface>
using __vtable_for _CCCL_NODEBUG_ALIAS = typename __overrides_for<_Interface>::__vtable;

///
/// __basic_vtable
///
template <class _Interface, auto... _Mbrs>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __basic_vtable
    : __rtti_base
    , __virtual_fn<_Mbrs>...
{
  using interface _CCCL_NODEBUG_ALIAS = _Interface;
  static constexpr size_t __cbases    = _CUDA_VSTD::__type_list_size<__unique_interfaces<interface>>::value;

  template <class _VPtr, class _Tp, auto... _OtherMembers, class... _Interfaces>
  _CUDAX_API constexpr __basic_vtable(_VPtr __vptr, overrides_for<_Tp, _OtherMembers...>, __tag<_Interfaces...>) noexcept
      : __rtti_base{__vtable_kind::__normal, __cbases, _CCCL_TYPEID(__basic_vtable)}
      , __virtual_fn<_Mbrs>{__override_tag<_Tp, _OtherMembers>{}}...
      , __vptr_map_{__base_vptr{__vptr->__query_interface(_Interfaces())}...}
  {}

  template <class _Tp, class _VPtr>
  _CUDAX_API constexpr __basic_vtable(__tag<_Tp>, _VPtr __vptr) noexcept
      : __basic_vtable{
          __vptr, __overrides_for<interface, _Tp>(), __unique_interfaces<interface, _CUDA_VSTD::__type_quote<__tag>>()}
  {}

  _CCCL_NODISCARD _CUDAX_API __vptr_for<interface> __query_interface(interface) const noexcept
  {
    return this;
  }

  template <class... _Others>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<__iset<_Others...>> __query_interface(__iset<_Others...>) const noexcept
  {
    using __remainder _CCCL_NODEBUG_ALIAS =
      _CUDA_VSTD::__type_list_size<_CUDA_VSTD::__type_find<__unique_interfaces<interface>, __iset<_Others...>>>;
    constexpr size_t __index = __cbases - __remainder::value;
    if constexpr (__index < __cbases)
    {
      // `_Interface` extends __iset<_Others...> exactly. We can return an actual
      // vtable pointer.
      return static_cast<__vtable_for<__iset<_Others...>> const*>(__vptr_map_[__index]);
    }
    else
    {
      // Otherwise, we have to return a subset vtable pointer, which does
      // dynamic interface lookup.
      return static_cast<__vptr_for<__iset<_Others...>>>(__query_interface(iunknown()));
    }
  }

  template <class _Other>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<_Other> __query_interface(_Other) const noexcept
  {
    constexpr size_t __index = __index_of<_Other, interface>::value;
    static_assert(__index < __cbases);
    return static_cast<__vptr_for<_Other>>(__vptr_map_[__index]);
  }

  __base_vptr __vptr_map_[__cbases];
};

///
/// __vtable implementation details
///

template <class... _Interfaces>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __vtable_tuple
    : __rtti_ex<sizeof...(_Interfaces)>
    , __vtable_for<_Interfaces>...
{
  static_assert((_CUDA_VSTD::is_class_v<_Interfaces> && ...), "expected class types");

  template <class _Tp, class _Super>
  _CUDAX_API constexpr __vtable_tuple(__tag<_Tp, _Super> __type) noexcept
      : __rtti_ex<sizeof...(_Interfaces)>{__type, __tag<_Interfaces...>(), this}
#ifdef _CCCL_COMPILER_MSVC
      // workaround for MSVC bug
      , __overrides_for<_Interfaces>::__vtable{__tag<_Tp>(), this}...
#else
      , __vtable_for<_Interfaces>{__tag<_Tp>(), this}...
#endif
  {
    static_assert(_CUDA_VSTD::is_class_v<_Super>, "expected a class type");
  }

  _LIBCUDACXX_TEMPLATE(class _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_included_in_v<_Interface, _Interfaces...>)
  _CCCL_NODISCARD _CUDAX_API constexpr __vptr_for<_Interface> __query_interface(_Interface) const noexcept
  {
    return static_cast<__vptr_for<_Interface>>(this);
  }
};

// The vtable type for type `_Interface` is a `__vtable_tuple` of `_Interface`
// and all of its base interfaces.
template <class _Interface>
using __vtable _CCCL_NODEBUG_ALIAS = __unique_interfaces<_Interface, _CUDA_VSTD::__type_quote<__vtable_tuple>>;

// __vtable_for_v<_Interface, _Tp> is an instance of `__vtable<_Interface>` that
// contains the overrides for `_Tp`.
template <class _Interface, class _Tp>
inline constexpr __vtable<_Interface> __vtable_for_v{__tag<_Tp, _Interface>()};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_VIRTUAL_TABLES_H
