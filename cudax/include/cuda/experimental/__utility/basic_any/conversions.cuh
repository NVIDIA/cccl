//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_CONVERSIONS_H
#define __CUDAX_DETAIL_BASIC_ANY_CONVERSIONS_H

#include <cuda/std/detail/__config>

#include "cuda/std/__cccl/attributes.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>

_CCCL_PUSH_MACROS
#undef interface

namespace cuda::experimental
{
//!
//! conversions
//!
//! Can one basic_any type convert to another? Implicitly? Explicitly?
//! Statically? Dynamically? We answer these questions by mapping two
//! cvref qualified basic_any types to archetype types, and then using
//! the built-in language rules to determine if the conversion is valid.
//!
struct __immovable_archetype
{
  __immovable_archetype()                             = default;
  __immovable_archetype(__immovable_archetype&&)      = delete;
  __immovable_archetype(const __immovable_archetype&) = delete;

  template <class _Value>
  _CUDAX_HOST_API __immovable_archetype(_Value) noexcept;
  template <class _Value>
  _CUDAX_HOST_API __immovable_archetype(_Value*) = delete;
};

struct __movable_archetype : __immovable_archetype
{
  __movable_archetype() = default;
  _CUDAX_HOST_API __movable_archetype(__movable_archetype&&) noexcept;
  __movable_archetype(const __movable_archetype&) = delete;
};

struct __copyable_archetype : __movable_archetype
{
  __copyable_archetype() = default;
  _CUDAX_HOST_API __copyable_archetype(__copyable_archetype const&);
};

template <class _Interface>
using __archetype_base =
  _CUDA_VSTD::__type_switch<(extension_of<_Interface, imovable<>> + extension_of<_Interface, icopyable<>>),
                            _CUDA_VSTD::__type_case<0, __immovable_archetype>,
                            _CUDA_VSTD::__type_case<1, __movable_archetype>,
                            _CUDA_VSTD::__type_case<2, __copyable_archetype>>;

template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface&&) -> __archetype_base<_Interface>;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface&) -> __archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface const&) -> __archetype_base<_Interface> const&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface*) -> __archetype_base<_Interface>*;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface const*) -> __archetype_base<_Interface> const*;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(__ireference<_Interface>) -> __archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(__ireference<_Interface const>) -> __archetype_base<_Interface> const&;

template <class _Archetype>
_CUDAX_HOST_API auto __as_immovable(_Archetype&&) -> __immovable_archetype;
template <class _Archetype>
_CUDAX_HOST_API auto __as_immovable(_Archetype&) -> __immovable_archetype&;
template <class _Archetype>
_CUDAX_HOST_API auto __as_immovable(_Archetype const&) -> __immovable_archetype const&;
template <class _Archetype>
_CUDAX_HOST_API auto __as_immovable(_Archetype*) -> __immovable_archetype*;
template <class _Archetype>
_CUDAX_HOST_API auto __as_immovable(_Archetype const*) -> __immovable_archetype const*;

#if _CCCL_COMPILER(MSVC)
// Strip top-level cv- and ref-qualifiers from pointer types:
template <class _Ty>
auto __normalize(_Ty&&) -> _Ty
{}
template <class _Ty>
auto __normalize(_Ty*) -> _Ty*
{}

template <class _Ty>
using __normalize_t = decltype(__cudax::__normalize(declval<_Ty>()));

template <class _Ty>
extern _CUDA_VSTD::__undefined<_Ty> __interface_from;
template <class _Interface>
extern __identity_t<_Interface (*)()> __interface_from<basic_any<_Interface>>;
template <class _Interface>
extern __identity_t<_Interface (*)()> __interface_from<basic_any<__ireference<_Interface>>>;
template <class _Interface>
extern __identity_t<_Interface& (*) ()> __interface_from<basic_any<_Interface>&>;
template <class _Interface>
extern __identity_t<_Interface const& (*) ()> __interface_from<basic_any<_Interface> const&>;
template <class _Interface>
extern __identity_t<_Interface* (*) ()> __interface_from<basic_any<_Interface>*>;
template <class _Interface>
extern __identity_t<_Interface const* (*) ()> __interface_from<basic_any<_Interface> const*>;
template <class _Interface>
extern __identity_t<_Interface* (*) ()> __interface_from<basic_any<__ireference<_Interface>>*>;
template <class _Interface>
extern __identity_t<_Interface* (*) ()> __interface_from<basic_any<__ireference<_Interface>> const*>;

template <class _CvAny>
using __normalized_interface_of _CCCL_NODEBUG_ALIAS = decltype(__interface_from<__normalize_t<_CvAny>>());

template <class _CvAny>
using __src_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_archetype(__interface_from<__normalize_t<_CvAny>>()));

template <class _CvAny>
using __dst_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_immovable(__cudax::__as_archetype(__interface_from<__normalize_t<_CvAny>>())));

#else // ^^^ MSVC ^^^ / vvv !MSVC vvv

template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<_Interface>&&) -> _Interface;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<__ireference<_Interface>>&&) -> _Interface;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<_Interface>&) -> _Interface&;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<_Interface> const&) -> _Interface const&;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<_Interface>*) -> _Interface*;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<_Interface> const*) -> _Interface const*;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<__ireference<_Interface>>*) -> _Interface*;
template <class _Interface>
_CUDAX_HOST_API auto __interface_from(basic_any<__ireference<_Interface>> const*) -> _Interface*;

template <class _CvAny>
using __normalized_interface_of _CCCL_NODEBUG_ALIAS = decltype(__cudax::__interface_from(declval<_CvAny>()));

template <class _CvAny>
using __src_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_archetype(__cudax::__interface_from(declval<_CvAny>())));

template <class _CvAny>
using __dst_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_immovable(__cudax::__as_archetype(__cudax::__interface_from(declval<_CvAny>()))));
#endif // !MSVC

// If the archetypes are implicitly convertible, then it is possible to
// dynamically cast from the source to the destination. The cast may fail,
// but at least it is possible.
template <class _SrcCvAny, class _DstCvAny>
_CCCL_CONCEPT __any_castable_to =
  _CUDA_VSTD::convertible_to<__src_archetype_of<_SrcCvAny>, __dst_archetype_of<_DstCvAny>>;

// If the archetypes are implicitly convertible **and** the source interface
// is an extension of the destination one, then it is possible to implicitly
// convert from the source to the destination.
template <class _SrcCvAny, class _DstCvAny>
_CCCL_CONCEPT __any_convertible_to =
  __any_castable_to<_SrcCvAny, _DstCvAny> && //
  extension_of<typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type,
               typename _CUDA_VSTD::remove_reference_t<_DstCvAny>::interface_type>;

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_CONVERSIONS_H
