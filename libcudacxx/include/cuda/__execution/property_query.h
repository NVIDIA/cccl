//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___EXECUTION_PROPERTY_QUERY_H
#define _CUDA___EXECUTION_PROPERTY_QUERY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/undefined.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

//! @brief Describes a query expression with a key and zero or more additional argument types.
//!
//! @tparam _Query The query key type.
//! @tparam _Args The types of the additional arguments passed to the query.
template <class _Query, class... _Args>
struct property_query
{};

//! @brief A list of query expressions advertised by an environment or query-providing object.
//!
//! A bare query key is shorthand for @c property_query<Query>. Entries with additional
//! argument types are represented with @c property_query<Query,Args...>.
//!
//! @tparam _Queries The advertised query expressions.
template <class... _Queries>
struct property_key_list
{};

//! @brief Whether a type is a specialization of @c property_key_list.
//!
//! @tparam _Tp The type to inspect.
template <class _Tp>
inline constexpr bool is_property_key_list_v = false;

template <class... _Queries>
inline constexpr bool is_property_key_list_v<property_key_list<_Queries...>> = true;

namespace __detail
{
template <class _Tp>
_CCCL_CONCEPT __has_reserved_property_keys = _CCCL_REQUIRES_EXPR(
  (_Tp))(typename(typename ::cuda::std::remove_cvref_t<_Tp>::__property_keys),
         requires(is_property_key_list_v<typename ::cuda::std::remove_cvref_t<_Tp>::__property_keys>));

template <class _Tp>
_CCCL_CONCEPT __has_public_property_keys = _CCCL_REQUIRES_EXPR(
  (_Tp))(typename(typename ::cuda::std::remove_cvref_t<_Tp>::property_keys),
         requires(is_property_key_list_v<typename ::cuda::std::remove_cvref_t<_Tp>::property_keys>));

template <class _Tp, bool = __has_reserved_property_keys<_Tp>, bool = __has_public_property_keys<_Tp>>
extern ::cuda::std::__undefined<_Tp>* __property_keys;

template <class _Tp, bool _HasPublicPropertyKeys>
extern typename ::cuda::std::remove_cvref_t<_Tp>::__property_keys __property_keys<_Tp, true, _HasPublicPropertyKeys>;

template <class _Tp>
extern typename ::cuda::std::remove_cvref_t<_Tp>::property_keys __property_keys<_Tp, false, true>;

template <class _Tp>
using __property_keys_t _CCCL_NODEBUG = decltype(__property_keys<_Tp>);
} // namespace __detail

//! @brief The advertised query list associated with a type.
//!
//! A user-defined type can customize this alias by defining
//! @c cuda::std::remove_cvref_t<_Tp>::property_keys as a @c property_key_list specialization. This alias
//! is not defined when @c _Tp has no discoverable property-key list.
//!
//! @tparam _Tp The type whose advertised query list is obtained.
template <class _Tp>
using property_keys_t _CCCL_NODEBUG =
  ::cuda::std::enable_if_t<is_property_key_list_v<__detail::__property_keys_t<_Tp>>, __detail::__property_keys_t<_Tp>>;

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___EXECUTION_PROPERTY_QUERY_H
