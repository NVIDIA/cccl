//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_GET_PROPERTY_H
#define _CUDA__MEMORY_RESOURCE_GET_PROPERTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__type_traits/remove_const_ref.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/__utility/declval.h>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/// \concept has_property
/// \brief The \c has_property concept
template <class _Resource, class _Property, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool has_property = false;

template <class _Resource, class _Property>
_LIBCUDACXX_INLINE_VAR constexpr bool has_property<
  _Resource,
  _Property,
  _CUDA_VSTD::void_t<decltype(get_property(_CUDA_VSTD::declval<const _Resource&>(), _CUDA_VSTD::declval<_Property>()))>> =
  true;

/// \concept property_with_value
/// \brief The \c property_with_value concept
template <class _Property>
using __property_value_t = typename _Property::value_type;

template <class _Property, class = void>
_LIBCUDACXX_INLINE_VAR constexpr bool property_with_value = false;

template <class _Property>
_LIBCUDACXX_INLINE_VAR constexpr bool property_with_value<_Property, _CUDA_VSTD::void_t<__property_value_t<_Property>>> =
  true;

/// \concept has_property_with
/// \brief The \c has_property_with concept
template <class _Resource, class _Property, class _Return>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __has_property_with_,
  requires(const _Resource& __res)(requires(property_with_value<_Property>),
                                   requires(_CUDA_VSTD::same_as<_Return, decltype(get_property(__res, _Property{}))>)));
template <class _Resource, class _Property, class _Return>
_LIBCUDACXX_CONCEPT has_property_with = _LIBCUDACXX_FRAGMENT(__has_property_with_, _Resource, _Property, _Return);

/// \concept __has_upstream_resource
/// \brief The \c __has_upstream_resource concept
template <class _Resource, class _Upstream>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __has_upstream_resource_,
  requires(const _Resource& __res)(
    requires(_CUDA_VSTD::same_as<_CUDA_VSTD::__remove_const_ref_t<decltype(__res.upstream_resource())>, _Upstream>)));
template <class _Resource, class _Upstream>
_LIBCUDACXX_CONCEPT __has_upstream_resource = _LIBCUDACXX_FRAGMENT(__has_upstream_resource_, _Resource, _Upstream);

/// class forward_property
/// \brief The \c forward_property crtp template simplifies the user facing side of forwarding properties
///        We can just derive from it to properly forward all properties
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__forward_property)
template <class _Derived, class _Upstream>
struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_TEMPLATE(class _Property)
  _LIBCUDACXX_REQUIRES((!property_with_value<_Property>) _LIBCUDACXX_AND has_property<_Upstream, _Property>)
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr void get_property(const _Derived&, _Property) noexcept {}

  // The indirection is needed, otherwise the compiler might believe that _Derived is an incomplete type
  _CCCL_EXEC_CHECK_DISABLE
  _LIBCUDACXX_TEMPLATE(class _Property, class _Derived2 = _Derived)
  _LIBCUDACXX_REQUIRES(property_with_value<_Property> _LIBCUDACXX_AND has_property<_Upstream, _Property> _LIBCUDACXX_AND
                         __has_upstream_resource<_Derived2, _Upstream>)
  _LIBCUDACXX_INLINE_VISIBILITY friend constexpr __property_value_t<_Property>
  get_property(const _Derived& __res, _Property __prop)
  {
    return get_property(__res.upstream_resource(), __prop);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

template <class _Derived, class _Upstream>
using forward_property = __forward_property::__fn<_Derived, _Upstream>;

/// class get_property
/// \brief The \c get_property customization point ensures that `cuda::get_property` is always available
_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__get_property)
void get_property();

struct __fn
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Upstream, class _Property>
  _LIBCUDACXX_INLINE_VISIBILITY constexpr auto operator()(const _Upstream& __res, _Property __prop) const
    noexcept(noexcept(get_property(__res, __prop))) -> decltype(get_property(__res, __prop))
  {
    return get_property(__res, __prop);
  }
};
_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_LIBCUDACXX_CPO_ACCESSIBILITY auto get_property = __get_property::__fn{};
} // namespace __cpo

_LIBCUDACXX_END_NAMESPACE_CUDA

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_GET_PROPERTY_H
