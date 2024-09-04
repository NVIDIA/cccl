//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/integer_sequence.h>

//! @file type_list.h
//! This file defines a type-list type and some fundamental algorithms on type
//! lists.
//!
//! It also defines a "meta-callable" protocol for parameterizing the type list
//! algorithms, and some higher-order meta-callables. A meta-callable is a class
//! type with a nested __apply template alias.
//!
//! For the purpose of this file, a "trait type" is a class type with a nested
//! type alias named \c type.

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document

template <class _Tp>
using __type = typename _Tp::type;

//! @brief Evaluate a meta-callable with the given arguments
template <class _Fn, class... _Ts>
using __type_call = typename _Fn::template __apply<_Ts...>;

//! @brief Turns a class or alias template into a meta-callable
template <template <class...> class _Fn>
struct __type_quote
{
  template <class... _Ts>
  using __apply = _Fn<_Ts...>;
};

//! @brief A meta-callable that composes two meta-callables
template <class _Fn1, class _Fn2>
struct __type_compose
{
  template <class... _Ts>
  using __apply = __type_call<_Fn1, __type_call<_Fn2, _Ts...>>;
};

//! @brief A meta-callable that binds the front arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_front
{
  template <class... _Us>
  using __apply = __type_call<_Fn, _Ts..., _Us...>;
};

//! @brief A meta-callable that binds the back arguments to a meta-callable
template <class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_back
{
  template <class... _Us>
  using __apply = __type_call<_Fn, _Us..., _Ts...>;
};

//! @brief A type list
template <class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_list
{
  static constexpr size_t const __size = sizeof...(_Ts);

  // A type_list behaves like a meta-callable
  // that takes a meta-callable and applies the
  // elements of the list to it.
  template <class _Fn, class... _Us>
  using __apply = __type_call<_Fn, _Ts..., _Us...>;
};

// Before the addition of inline variables, it was necessary to
// provide a definition for constexpr class static data members.
#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <class... _Ts>
constexpr size_t const __type_list<_Ts...>::__size;
#  endif

//! @brief Given a type list and a list of types, append the types to the list.
template <class _List, class... _Ts>
using __type_push_back = __type_call<_List, __type_quote<__type_list>, _Ts...>;

//! @brief Given a type list and a list of types, prepend the types to the list.
template <class _List, class... _Ts>
using __type_push_front = __type_call<__type_list<__type_quote<__type_list>, _Ts...>, _List>;

namespace detail
{
template <class _Fn, class _ArgList, class _Enable = void>
struct __type_callable
{
  using type = _CUDA_VSTD::false_type;
};

template <class _Fn, class... _Ts>
struct __type_callable<_Fn, __type_list<_Ts...>, _CUDA_VSTD::void_t<__type_call<_Fn, _Ts...>>>
{
  using type = _CUDA_VSTD::true_type;
};
} // namespace detail

//! @brief Test whether a meta-callable is callable with a given set of
//! arguments.
//!
//! Given a meta-callable and a list of arguments, return \c true_type if
//! the meta-callable can be called with the arguments, and \c false_type
//! otherwise.
template <class _Fn, class... _Ts>
using __type_callable = __type<detail::__type_callable<_Fn, __type_list<_Ts...>>>;

namespace detail
{
template <class _Fn, class _ArgList, class _Enable = void>
struct __type_defer
{};

template <class _Fn, class... _Ts>
struct __type_defer<_Fn, __type_list<_Ts...>, _CUDA_VSTD::void_t<__type_call<_Fn, _Ts...>>>
{
  using type = __type_call<_Fn, _Ts...>;
};
} // namespace detail

//! @brief Defer the evaluation of a meta-callable with a list of arguments.
//!
//! Given a meta-callable and a list of arguments, return a trait type \c T
//! where \c T::type is the result of evaluating the meta-callable with the
//! arguments, or if the meta-callable is not callable with the arguments, a
//! class type without a nested \c ::type type alias.
template <class _Fn, class... _Ts>
struct __type_defer : detail::__type_defer<_Fn, __type_list<_Ts...>>
{};

// Implementation for indexing into a list of types:
#  if defined(__cpp_pack_indexing)

template <size_t _Ip, class... _Ts>
using __type_index_c = _Ts...[_Ip];

template <class _Ip, class... _Ts>
using __type_index = _Ts...[_Ip::value];

#  elif __has_builtin(__type_pack_element)

namespace detail
{
// On some versions of gcc, __type_pack_element cannot be mangled so
// hide it behind a named template.
template <size_t _Ip>
struct __type_index_fn
{
  template <class... _Ts>
  using __apply = __type_pack_element<_Ip, _Ts...>;
};
} // namespace detail

template <size_t _Ip, class... _Ts>
using __type_index_c = __type_call<detail::__type_index_fn<_Ip>, _Ts...>;

template <class _Ip, class... _Ts>
using __type_index = __type_call<detail::__type_index_fn<_Ip::value>, _Ts...>;

#  else

namespace detail
{
template <size_t>
using __void_ptr = void*;

template <class _Ty>
using __type_ptr = _CUDA_VSTD::type_identity<_Ty>*;

template <size_t _Ip, class _Ty = _CUDA_VSTD::make_index_sequence<_Ip>>
struct __type_index_fn;

template <size_t _Ip, size_t... _Is>
struct __type_index_fn<_Ip, _CUDA_VSTD::index_sequence<_Is...>>
{
  template <class _Up>
  static _Up __apply_(__void_ptr<_Is>..., _Up*, ...);

  template <class... _Ts>
  using __apply = __type<decltype(__type_index_fn::__apply_(__type_ptr<_Ts>()...))>;
};
} // namespace detail

template <size_t _Ip, class... _Ts>
using __type_index_c = __type_call<detail::__type_index_fn<_Ip>, _Ts...>;

template <class _Ip, class... _Ts>
using __type_index = __type_call<detail::__type_index_fn<_Ip::value>, _Ts...>;

#  endif

namespace detail
{
template <size_t _Ip>
struct __type_at_fn
{
  template <class... _Ts>
  using __apply = __type_index_c<_Ip, _Ts...>;
};
} // namespace detail

//! @brief Given a type list and an index, return the type at that index.
template <size_t _Ip, class _List>
using __type_at_c = __type_call<_List, detail::__type_at_fn<_Ip>>;

//! @brief Given a type list and an index, return the type at that index.
template <class _Ip, class _List>
using __type_at = __type_call<_List, detail::__type_at_fn<_Ip::value>>;

//! @brief Given a type list return the type at the front of the list.
template <class _List>
using __type_front = __type_at_c<0, _List>;

//! @brief Given a type list return the type at the back of the list.
template <class _List>
using __type_back = __type_at_c<_List::__size - 1, _List>;

namespace detail
{
template <bool _Empty>
struct __type_maybe_concat_fn
{
  template <class... _Ts, class... _As, class... _Bs, class... _Cs, class... _Ds, class... _Tail>
  static auto
  __fn(__type_list<_Ts...>*,
       __type_list<_As...>*,
       __type_list<_Bs...>* = nullptr,
       __type_list<_Cs...>* = nullptr,
       __type_list<_Ds...>* = nullptr,
       _Tail*... __tail)
    -> decltype(__type_maybe_concat_fn<(sizeof...(_Tail) == 0)>::__fn(
      static_cast<__type_list<_Ts..., _As..., _Bs..., _Cs..., _Ds...>*>(nullptr), __tail...));
};

template <>
struct __type_maybe_concat_fn<true>
{
  template <class... _As>
  static auto __fn(__type_list<_As...>*) -> __type_list<_As...>;
};

struct __type_concat_fn
{
  template <class... _Lists>
  using __apply =
    decltype(__type_maybe_concat_fn<(sizeof...(_Lists) == 0)>::__fn({}, static_cast<_Lists*>(nullptr)...));
};
} // namespace detail

//! @brief Concatenate a list of type lists into a single type list.
//!
//! When passed no type lists, \c __type_concat returns an empty type list.
template <class... _Lists>
using __type_concat = __type_call<detail::__type_concat_fn, _Lists...>;

namespace detail
{
template <bool _IsEmpty>
struct __type_maybe_find_if_fn // Type list is not empty
{
  template <class _Fn, class _Head, class... _Tail>
  using __apply =
    _CUDA_VSTD::_If<__type_call<_Fn, _Head>::value,
                    __type_list<_Head, _Tail...>,
                    __type_call<__type_maybe_find_if_fn<sizeof...(_Tail) == 0>, _Fn, _Tail...>>;
};

template <>
struct __type_maybe_find_if_fn<true> // Type list is empty
{
  template <class, class... _None>
  using __apply = __type_list<>;
};

template <class _Fn>
struct __type_find_if_fn
{
  template <class... _Ts>
  using __apply = __type_call<__type_maybe_find_if_fn<sizeof...(_Ts) == 0>, _Fn, _Ts...>;
};
} // namespace detail

//! @brief Given a type list and a predicate, find the first type in a list that
//! satisfies a predicate. It returns a type list containing the first type that
//! satisfies the predicate and all the types after it.
//!
//! If no type in the list satisfies the predicate, \c __type_find_if
//! returns an empty type list.
template <class _List, class _Fn>
using __type_find_if = __type_call<_List, detail::__type_find_if_fn<_Fn>>;

namespace detail
{
template <class _Fn>
struct __type_transform_fn
{
  template <class... _Ts>
  using __apply = __type_list<__type_call<_Fn, _Ts>...>;
};
} // namespace detail

//! @brief Given a type list and a unary meta-callable, apply the meta-callable
//! to each type in the list. It returns a new type list containing the results.
template <class _List, class _Fn>
using __type_transform = __type_call<_List, detail::__type_transform_fn<_Fn>>;

//
// Implementation for folding a type list either left or right
//
namespace detail
{
template <bool>
struct __type_maybe_fold_right_fn
{
  template <class _Fn, class _State, class _Head, class... _Tail>
  using __apply =
    __type_call<__type_maybe_fold_right_fn<sizeof...(_Tail) == 0>, _Fn, __type_call<_Fn, _State, _Head>, _Tail...>;
};

template <>
struct __type_maybe_fold_right_fn<true> // empty pack
{
  template <class _Fn, class _State, class...>
  using __apply = _State;
};

template <class _Init, class _Fn>
struct __type_fold_right_fn
{
  template <class... _Ts>
  using __apply = __type_call<__type_maybe_fold_right_fn<sizeof...(_Ts) == 0>, _Fn, _Init, _Ts...>;
};

template <bool>
struct __type_maybe_fold_left_fn
{
  template <class _Fn, class _State, class _Head, class... _Tail>
  using __apply =
    __type_call<_Fn, __type_call<__type_maybe_fold_left_fn<sizeof...(_Tail) == 0>, _Fn, _State, _Tail...>, _Head>;
};

template <>
struct __type_maybe_fold_left_fn<true> // empty pack
{
  template <class _Fn, class _State, class...>
  using __apply = _State;
};

template <class _Init, class _Fn>
struct __type_fold_left_fn
{
  template <class... _Ts>
  using __apply = __type_call<__type_maybe_fold_left_fn<sizeof...(_Ts) == 0>, _Fn, _Init, _Ts...>;
};

} // namespace detail

//! @brief Fold a type list from the right with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_right = __type_call<_List, detail::__type_fold_right_fn<_Init, _Fn>>;

//! @brief Fold a type list from the left with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_left = __type_call<_List, detail::__type_fold_left_fn<_Init, _Fn>>;

// A unary meta-callable for converting a type to its size in bytes
struct __type_sizeof
{
  template <class _Ty>
  using __apply = _CUDA_VSTD::integral_constant<size_t, sizeof(_Ty)>;
};

//! @brief Perform a logical AND operation on a list of Boolean types.
//!
//! @note The AND operation is not short-circuiting.
struct __type_strict_and
{
#  ifndef _LIBCUDACXX_HAS_NO_FOLD_EXPRESSIONS
  template <class... _Ts>
  using __apply = _CUDA_VSTD::bool_constant<(_Ts::value && ...)>;
#  else
  template <class... _Ts>
  using __apply =
    _CUDA_VSTD::bool_constant<_CCCL_TRAIT(_CUDA_VSTD::is_same,
                                          _CUDA_VSTD::integer_sequence<bool, true, _Ts::value...>,
                                          _CUDA_VSTD::integer_sequence<bool, _Ts::value..., true>)>;
#  endif
};

//! @brief Perform a logical OR operation on a list of Boolean types.
//!
//! @note The OR operation is not short-circuiting.
struct __type_strict_or
{
#  ifndef _LIBCUDACXX_HAS_NO_FOLD_EXPRESSIONS
  template <class... _Ts>
  using __apply = _CUDA_VSTD::bool_constant<(_Ts::value || ...)>;
#  else
  template <class... _Ts>
  using __apply =
    _CUDA_VSTD::bool_constant<!_CCCL_TRAIT(_CUDA_VSTD::is_same,
                                           _CUDA_VSTD::integer_sequence<bool, false, _Ts::value...>,
                                           _CUDA_VSTD::integer_sequence<bool, _Ts::value..., false>)>;
#  endif
};

//! @brief Perform a logical NOT operation on a Boolean type.
//!
//! @note The AND operation is not short-circuiting.
struct __type_not
{
  template <class _Ty>
  using __apply = _CUDA_VSTD::bool_constant<(!_Ty::value)>;
};

//! @brief Test whether two integral constants are equal.
struct __type_equal
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value == _Uy::value)>;
};

//! @brief Test whether two integral constants are not equal.
struct __type_not_equal
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value != _Uy::value)>;
};

//! @brief Test whether one integral constant is less than another.
struct __type_less
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value < _Uy::value)>;
};

//! @brief Test whether one integral constant is less than or equal to another.
struct __type_less_equal
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value <= _Uy::value)>;
};

//! @brief Test whether one integral constant is greater than another.
struct __type_greater
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value > _Uy::value)>;
};

//! @brief Test whether one integral constant is greater than or equal to another.
struct __type_greater_equal
{
  template <class _Ty, class _Uy>
  using __apply = _CUDA_VSTD::bool_constant<(_Ty::value >= _Uy::value)>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_LIST_H
