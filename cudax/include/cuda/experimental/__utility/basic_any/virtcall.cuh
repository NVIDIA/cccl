//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_VIRTCALL_H
#define __CUDAX_DETAIL_BASIC_ANY_VIRTCALL_H

#include <cuda/std/detail/__config>

#include "cuda/std/__cccl/unreachable.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/__concept_macros.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_from.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/virtual_functions.cuh>

namespace cuda::experimental
{
///
/// __virtuals_map
///

/// The virtuals map is an extra convenience for interface authors. To make a
/// virtual function call, the user must provide the member function pointer
/// corresponding to the virtual, as in:
///
/// @code
/// template <class...>
/// struct ifoo {
///   void meow(auto... __args) {
///     // dispatch to the &ifoo<>::meow virtual function
///     // NB: the `<>` after `ifoo` is significant!
///     virtcall<&ifoo<>::meow>(this, __args...);
///     //            ^^
///   }
///  ...
/// };
/// @endcode
///
/// When taking the address of the member, it is very easy to forget the `<>`
/// after the interface name, which would result in a compilation error --
/// except for the virtuals map, which substitutes the correct member function
/// pointer for the user so they don't have to think about it.
template <auto _Mbr, auto _BoundMbr>
struct __virtuals_map_pair
{
  // map ifoo<>::meow to itself
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API constexpr auto operator()(__ctag<_Mbr>) const noexcept
  {
    return _Mbr;
  }

  // map ifoo<_Super>::meow to ifoo<>::meow
  _CCCL_NODISCARD _CUDAX_TRIVIAL_HOST_API constexpr auto operator()(__ctag<_BoundMbr>) const noexcept
  {
    return _Mbr;
  }
};

template <class, class>
struct __virtuals_map;

template <class _Interface, auto... _Mbrs, class _BoundInterface, auto... _BoundMbrs>
struct __virtuals_map<overrides_for<_Interface, _Mbrs...>, overrides_for<_BoundInterface, _BoundMbrs...>>
    : __virtuals_map_pair<_Mbrs, _BoundMbrs>...
{
  using __virtuals_map_pair<_Mbrs, _BoundMbrs>::operator()...;
};

template <class _Interface, class _Super>
using __virtuals_map_for _CCCL_NODEBUG_ALIAS =
  __virtuals_map<__overrides_for<_Interface>, __overrides_for<__rebind_interface<_Interface, _Super>>>;

///
/// virtcall
///

// If the interface is __ireference<MyInterface const>, then calls to non-const
// member functions are not allowed.
template <auto, class... _Interface>
inline constexpr bool __valid_virtcall = sizeof...(_Interface) == 1;

template <auto _Mbr, class _Interface>
inline constexpr bool __valid_virtcall<_Mbr, __ireference<_Interface const>> = __virtual_fn<_Mbr>::__const_fn;

template <auto _Mbr, class _Interface, class _Super, class _Self, class... _Args>
_CUDAX_HOST_API auto __virtcall(_Self* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  auto* __vptr = __basic_any_access::__get_vptr(*__self)->__query_interface(_Interface());
  auto* __obj  = __basic_any_access::__get_optr(*__self);
  // map the member function pointer to the correct one if necessary
  constexpr auto _Mbr2 = __virtuals_map_for<_Interface, _Super>{}(__ctag<_Mbr>());
  return __vptr->__virtual_fn<_Mbr2>::__fn_(__obj, static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class _Super, class... _Args)
_LIBCUDACXX_REQUIRES(__valid_virtcall<_Mbr, _Super>)
_CUDAX_TRIVIAL_HOST_API auto virtcall(_Interface<_Super>* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  return __cudax::__virtcall<_Mbr, _Interface<>, _Super>(
    __cudax::basic_any_from(__self), static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class _Super, class... _Args)
_LIBCUDACXX_REQUIRES(__valid_virtcall<_Mbr, _Super>)
_CUDAX_TRIVIAL_HOST_API auto virtcall(_Interface<_Super> const* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  return __cudax::__virtcall<_Mbr, _Interface<>, _Super>(
    __cudax::basic_any_from(__self), static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class... _Super, class... _Args)
_LIBCUDACXX_REQUIRES((!__valid_virtcall<_Mbr, _Super...>) )
_CUDAX_TRIVIAL_HOST_API auto virtcall(_Interface<_Super...> const*, _Args&&...) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  constexpr bool __const_correct_virtcall = __valid_virtcall<_Mbr, _Super...> || sizeof...(_Super) == 0;
  // If this static assert fires, then you have called a non-const member
  // function on a `basic_any<I const&>`. This would violate const-correctness.
  static_assert(__const_correct_virtcall, "This function call is not const correct.");
  // This overload can also be selected when called from the thunks of
  // unspecialized interfaces. Those thunks should never be called, but they
  // must exist to satisfy the compiler.
  _CCCL_UNREACHABLE();
}

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_VIRTCALL_H
