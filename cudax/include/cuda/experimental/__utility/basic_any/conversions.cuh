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

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>

namespace cuda::experimental
{
///
/// conversions
///
/// Can one basic_any type convert to another? Implicitly? Explicitly?
/// Statically? Dynamically? We answer these questions by mapping two
/// cvref qualified basic_any types to archetype types, and then using
/// the built-in language rules to determine if the conversion is valid.
///
struct __immovable_archetype
{
  __immovable_archetype()                        = default;
  __immovable_archetype(__immovable_archetype&&) = delete;

  template <class _Value>
  _CUDAX_HOST_API __immovable_archetype(_Value) noexcept;
  template <class _Value>
  _CUDAX_HOST_API __immovable_archetype(_Value*) = delete;
};

struct __movable_archetype : __immovable_archetype
{
  __movable_archetype() = default;
  _CUDAX_HOST_API __movable_archetype(__movable_archetype&&) noexcept;
};

struct __copyable_archetype : __movable_archetype
{
  __copyable_archetype() = default;
  _CUDAX_HOST_API __copyable_archetype(__copyable_archetype const&);
};

template <class _Interface>
using _archetype_base = _CUDA_VSTD::conditional_t<
  extension_of<_Interface, icopyable<>>,
  __copyable_archetype,
  _CUDA_VSTD::conditional_t<extension_of<_Interface, imovable<>>, __movable_archetype, __immovable_archetype>>;

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

template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface&&) -> _archetype_base<_Interface>;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface&) -> _archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface const&) -> _archetype_base<_Interface> const&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface*) -> _archetype_base<_Interface>*;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(_Interface const*) -> _archetype_base<_Interface> const*;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(__ireference<_Interface>) -> _archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_HOST_API auto __as_archetype(__ireference<_Interface const>) -> _archetype_base<_Interface> const&;

template <class _Interface>
_CUDAX_HOST_API auto __as_immovable(_Interface&&) -> __immovable_archetype;
template <class _Interface>
_CUDAX_HOST_API auto __as_immovable(_Interface&) -> __immovable_archetype&;
template <class _Interface>
_CUDAX_HOST_API auto __as_immovable(_Interface const&) -> __immovable_archetype const&;
template <class _Interface>
_CUDAX_HOST_API auto __as_immovable(_Interface*) -> __immovable_archetype*;
template <class _Interface>
_CUDAX_HOST_API auto __as_immovable(_Interface const*) -> __immovable_archetype const*;

template <class CvAny>
using __normalized_interface_of _CCCL_NODEBUG_ALIAS = decltype(__cudax::__interface_from(declval<CvAny>()));

template <class CvAny>
using __src_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_archetype(__cudax::__interface_from(declval<CvAny>())));

template <class CvAny>
using __dst_archetype_of _CCCL_NODEBUG_ALIAS =
  decltype(__cudax::__as_immovable(__cudax::__as_archetype(__cudax::__interface_from(declval<CvAny>()))));

// If the archetypes are implicitly convertible, then it is possible to
// dynamically cast from the source to the destination. The cast may fail,
// but at least it is possible.
template <class _SrcCvAny, class _DstCvAny>
_LIBCUDACXX_CONCEPT __any_castable_to =
  _CUDA_VSTD::is_convertible_v<__src_archetype_of<_SrcCvAny>, __dst_archetype_of<_DstCvAny>>;

// If the archetypes are implicitly convertible **and** the source interface
// is an extension of the destination one, then it is possible to implicitly
// convert from the source to the destination.
template <class _SrcCvAny, class _DstCvAny>
_LIBCUDACXX_CONCEPT __any_convertible_to =
  __any_castable_to<_SrcCvAny, _DstCvAny> && //
  extension_of<typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type,
               typename _CUDA_VSTD::remove_reference_t<_DstCvAny>::interface_type>;

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_CONVERSIONS_H
