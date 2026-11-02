//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_TYPE_LIST_H
#define _CUDA_STD___TYPE_TRAITS_TYPE_LIST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

//! \file type_list.h
//! This file defines a type-list type and the most commonly used algorithms on
//! type lists. Less commonly used algorithms are defined in
//! type_list_operations.h.
//!
//! It also defines a "meta-callable" protocol for parameterizing the type list
//! algorithms. A meta-callable is a class type with a nested \c __call template
//! alias.
//!
//! For the purpose of this file, a "trait type" is a class type with a nested
//! type alias named \c type.

#if !defined(_CCCL_META_UNROLL_LIMIT)
#  if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVHPC)
#    define _CCCL_META_UNROLL_LIMIT 10
#  else
#    define _CCCL_META_UNROLL_LIMIT 16
#  endif
#endif

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

template <class... _Ts>
struct __type_list;

template <class _Ty>
using __type = typename _Ty::type;

//! \brief Evaluate a meta-callable with the given arguments
template <class _Fn, class... _Ts>
using __type_call = typename _Fn::template __call<_Ts...>;

//! \brief Evaluate a unary meta-callable with the given argument
template <class _Fn, class _Ty>
using __type_call1 = typename _Fn::template __call<_Ty>;

namespace __detail
{
template <size_t _DependentValue>
struct __type_call_indirect_fn
{
  template <template <class...> class _Fn, class... _Ts>
  using __call _CCCL_NODEBUG = _Fn<_Ts...>;
};
} // namespace __detail

//! \brief Evaluate a meta-callable with the given arguments, with an indirection
//! to avoid the dreaded "pack expansion argument for non-pack parameter" error.
template <class _Fn, class... _Ts>
using __type_call_indirect = //
  typename __detail::__type_call_indirect_fn<sizeof(
    __type_list<_Fn, _Ts...>*)>::template __call<_Fn::template __call, _Ts...>;

//! \brief Turns a class or alias template into a meta-callable
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = _Fn<_Ts...>;
};

//! \brief A type list
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_list
{
  static constexpr size_t const __size = sizeof...(_Ts);

  // A type_list behaves like a meta-callable
  // that takes a meta-callable and applies the
  // elements of the list to it.
  template <class _Fn, class... _Us>
  using __call _CCCL_NODEBUG = __type_call<_Fn, _Ts..., _Us...>;
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
template <class... _Ts>
constexpr size_t const __type_list<_Ts...>::__size;

//! \brief Given a type list and a list of types, append the types to the list.
template <class _List, class... _Ts>
using __type_push_back = __type_call<_List, __type_quote<__type_list>, _Ts...>;

// Implementation for indexing into a list of types:
#  if _CCCL_HAS_PACK_INDEXING()

template <size_t _Ip, class... _Ts>
using __type_index_c = _Ts...[_Ip];

// Versions of nvcc prior to 12.0 have trouble with pack expansion into
// __type_pack_element in an alias template, so we use the fall-back
// implementation instead.
#  elif defined(_CCCL_BUILTIN_TYPE_PACK_ELEMENT)

namespace __detail
{
// On some versions of gcc, __type_pack_element cannot be mangled so
// hide it behind a named template.
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_index_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = _CCCL_BUILTIN_TYPE_PACK_ELEMENT(_Ip, _Ts...);
};
} // namespace __detail

template <size_t _Ip, class... _Ts>
using __type_index_c = __type_call<__detail::__type_index_fn<_Ip>, _Ts...>;

#  else // ^^^ _CCCL_BUILTIN_TYPE_PACK_ELEMENT ^^^ / vvv !_CCCL_BUILTIN_TYPE_PACK_ELEMENT vvv

// Fallback implementation for __type_index_c uses multiple inheritance. See:
// https://ldionne.com/2015/11/29/efficient-parameter-pack-indexing/

namespace __detail
{
template <size_t _Ip, class _Ty>
struct __type_tuple_elem
{
  using type _CCCL_NODEBUG = _Ty;
};

template <class _Is, class... _Ts>
struct __type_tupl;

template <size_t... _Is, class... _Ts>
struct __type_tupl<index_sequence<_Is...>, _Ts...> : __type_tuple_elem<_Is, _Ts>...
{};

template <class... _Ts>
using __type_tuple = __type_tupl<make_index_sequence<sizeof...(_Ts)>, _Ts...>;

template <size_t _Ip, class _Ty>
_CCCL_API inline __type_tuple_elem<_Ip, _Ty> __type_tuple_get(__type_tuple_elem<_Ip, _Ty>);

template <size_t _Ip, class... _Ts>
using __type_tuple_element_t _CCCL_NODEBUG = __type<decltype(__detail::__type_tuple_get<_Ip>(__type_tuple<_Ts...>{}))>;

struct __type_index_large_size_fn
{
  template <class _Ip, class... _Ts>
  using __call _CCCL_NODEBUG = __type_tuple_element_t<_Ip::value, _Ts...>;
};

template <size_t _Ip>
struct __type_index_small_size_fn;

#    define _M0(X) class,
#    define _M1(_N)                                             \
      template <>                                               \
      struct __type_index_small_size_fn<_N>                     \
      {                                                         \
        template <_CCCL_PP_REPEAT(_N, _M0) class _Ty, class...> \
        using __call _CCCL_NODEBUG = _Ty;                       \
      };

_CCCL_PP_REPEAT_REVERSE(_CCCL_META_UNROLL_LIMIT, _M1)

#    undef _M0
#    undef _M1

template <bool _IsSmall>
struct __type_index_select_fn : __type_index_large_size_fn // Default for larger indices
{};

template <>
struct __type_index_select_fn<true> // Fast implementation for smaller indices
{
  template <class _Ip, class... _Ts>
  using __call _CCCL_NODEBUG = __type_call_indirect<__type_index_small_size_fn<_Ip::value>, _Ts...>;
};
} // namespace __detail

#    if !_CCCL_COMPILER(MSVC)

template <size_t _Ip, class... _Ts>
using __type_index_c =
  __type_call<__detail::__type_index_select_fn<(_Ip < _CCCL_META_UNROLL_LIMIT)>, integral_constant<size_t, _Ip>, _Ts...>;

#    else // ^^^ !_CCCL_COMPILER(MSVC) ^^^ / vvv _CCCL_COMPILER(MSVC) vvv

// Simplify the implementation for MSVC, which has trouble with the above
template <size_t _Ip, class... _Ts>
using __type_index_c = __detail::__type_index_large_size_fn::__call<integral_constant<size_t, _Ip>, _Ts...>;

#    endif // !_CCCL_COMPILER(MSVC)

#  endif // !_CCCL_BUILTIN_TYPE_PACK_ELEMENT

namespace __detail
{
template <size_t _Ip>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_at_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __type_index_c<_Ip, _Ts...>;
};
} // namespace __detail

//! \brief Given a type list and an index, return the type at that index.
template <size_t _Ip, class _List>
using __type_at_c = __type_call1<_List, __detail::__type_at_fn<_Ip>>;

//! \brief Given a type list return the type at the front of the list.
template <class _List>
using __type_front = __type_at_c<0, _List>;

namespace __detail
{
template <bool _IsEmpty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn;

template <bool _Found>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_find_if_found
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG = __type_list<_Head, _Tail...>;
};

template <>
struct __type_find_if_found<false>
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG = __type_call<__type_maybe_find_if_fn<sizeof...(_Tail) == 0>, _Fn, _Tail...>;
};

template <bool _IsEmpty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn // Type list is not empty
{
  template <class _Fn, class _Head, class... _Tail>
  using __call _CCCL_NODEBUG = __type_call<__type_find_if_found<__type_call1<_Fn, _Head>::value>, _Fn, _Head, _Tail...>;
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_find_if_fn<true> // Type list is empty
{
  template <class, class... _None>
  using __call _CCCL_NODEBUG = __type_list<>;
};

template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_find_if_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __type_call<__type_maybe_find_if_fn<sizeof...(_Ts) == 0>, _Fn, _Ts...>;
};
} // namespace __detail

//! \brief Given a type list and a predicate, find the first type in the list
//! that satisfies the predicate. It returns a type list containing the first
//! type that satisfies the predicate and all the types after it.
//!
//! If no type in the list satisfies the predicate, \c __type_find_if returns an
//! empty type list.
template <class _List, class _Fn>
using __type_find_if = __type_call1<_List, __detail::__type_find_if_fn<_Fn>>;

#endif // _CCCL_DOXYGEN_INVOKED

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_TYPE_LIST_H
