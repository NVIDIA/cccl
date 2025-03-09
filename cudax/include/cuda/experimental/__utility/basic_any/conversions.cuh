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
template <bool _Movable, bool _Copyable>
struct __archetype;

// Archetype for interfaces that extend neither imovable nor icopyable
template <>
struct __archetype<false, false> // immovable archetype
{
  __archetype()                   = default;
  __archetype(__archetype&&)      = delete;
  __archetype(const __archetype&) = delete;

  template <class _Value>
  _CUDAX_HOST_API __archetype(_Value) noexcept;
  template <class _Value>
  _CUDAX_HOST_API __archetype(_Value*) = delete;
};

// Archetype for interfaces that extend imovable but not icopyable
template <>
struct __archetype<true, false> : __archetype<false, false> // movable archetype
{
  __archetype() = default;
  _CUDAX_HOST_API __archetype(__archetype&&) noexcept;
  __archetype(const __archetype&) = delete;
};

// Archetype for interfaces that extend icopyable
template <>
struct __archetype<true, true> : __archetype<true, false>
{
  __archetype() = default;
  _CUDAX_HOST_API __archetype(__archetype const&);
};

template <class _Interface>
using __archetype_t _CCCL_NODEBUG_ALIAS =
  __archetype<extension_of<_Interface, imovable<>>, extension_of<_Interface, icopyable<>>>;

// Strip top-level cv- and ref-qualifiers from pointer types:
template <class _Ty>
auto __normalize(_Ty&&) -> _Ty
{}
template <class _Ty>
auto __normalize(_Ty*) -> _Ty*
{}

template <class _Ty>
using __normalize_t _CCCL_NODEBUG_ALIAS = decltype(__cudax::__normalize(declval<_Ty>()));

// Used to map a basic_any specialization to a normalized interface type:
template <class _Ty>
extern _CUDA_VSTD::__undefined<_Ty> __interface_from;
template <class _Interface>
extern _Interface __interface_from<basic_any<_Interface>>;
template <class _Interface>
extern _Interface __interface_from<basic_any<__ireference<_Interface>>>;
template <class _Interface>
extern _Interface& __interface_from<basic_any<_Interface>&>;
template <class _Interface>
extern _Interface const& __interface_from<basic_any<_Interface> const&>;
template <class _Interface>
extern _Interface* __interface_from<basic_any<_Interface>*>;
template <class _Interface>
extern _Interface const* __interface_from<basic_any<_Interface> const*>;
template <class _Interface>
extern _Interface* __interface_from<basic_any<__ireference<_Interface>>*>;
template <class _Interface>
extern _Interface* __interface_from<basic_any<__ireference<_Interface>> const*>;

// Used to map a normalized interface type to an archetype for conversion testing:
template <class _Interface>
extern __archetype_t<_Interface> __as_archetype;
template <class _Interface>
extern __archetype_t<_Interface>& __as_archetype<_Interface&>;
template <class _Interface>
extern __archetype_t<_Interface> const& __as_archetype<_Interface const&>;
template <class _Interface>
extern __archetype_t<_Interface>* __as_archetype<_Interface*>;
template <class _Interface>
extern __archetype_t<_Interface> const* __as_archetype<_Interface const*>;
template <class _Interface>
extern __archetype_t<_Interface>& __as_archetype<__ireference<_Interface>>;
template <class _Interface>
extern __archetype_t<_Interface> const& __as_archetype<__ireference<_Interface const>>;

// Used to map an archetype to an immovable archetype
template <class _Archetype>
extern __archetype<false, false> __as_immovable;
template <class _Archetype>
extern __archetype<false, false>& __as_immovable<_Archetype&>;
template <class _Archetype>
extern __archetype<false, false> const& __as_immovable<_Archetype const&>;
template <class _Archetype>
extern __archetype<false, false>* __as_immovable<_Archetype*>;
template <class _Archetype>
extern __archetype<false, false> const* __as_immovable<_Archetype const*>;

template <class _CvAny>
using __normalized_interface_of _CCCL_NODEBUG_ALIAS = __normalize_t<decltype(__interface_from<__normalize_t<_CvAny>>)>;

template <class _CvAny>
using __src_archetype_of _CCCL_NODEBUG_ALIAS = decltype(__as_archetype<__normalized_interface_of<_CvAny>>);

template <class _CvAny>
using __dst_archetype_of _CCCL_NODEBUG_ALIAS = decltype(__as_immovable<__src_archetype_of<_CvAny>>);

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
