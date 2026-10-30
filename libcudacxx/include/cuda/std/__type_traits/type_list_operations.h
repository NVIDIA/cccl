//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_TYPE_LIST_OPERATIONS_H
#define _CUDA_STD___TYPE_TRAITS_TYPE_LIST_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/undefined.h>

//! \file type_list_operations.h
//! This file defines higher-order meta-callables and the less commonly used
//! algorithms on type lists. The type list type itself and the most commonly
//! used algorithms are defined in type_list.h.

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

//! \brief Turns a unary class or alias template into a meta-callable
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote1
{
  template <class _Ty>
  using __call _CCCL_NODEBUG = _Fn<_Ty>;
};

//! \brief Turns a unary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait1 such that \c __type_quote_trait1<_Fn>::__call<_Ty> is
//! equivalent to \c _Fn<_Ty>::type.
template <template <class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait1
{
  template <class _Ty>
  using __call _CCCL_NODEBUG = __type<_Fn<_Ty>>;
};

//! \brief Turns a binary trait class template \c _Fn into a meta-callable
//! \c __type_quote_trait2 such that \c __type_quote_trait2<_Fn>::__call<_Ty,_Uy>
//! is equivalent to \c _Fn<_Ty,_Uy>::type.
template <template <class, class> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_quote_trait2
{
  template <class _Ty, class _Uy>
  using __call _CCCL_NODEBUG = __type<_Fn<_Ty, _Uy>>;
};

//! \brief Turns a class or alias template into a meta-callable with an
//! indirection to avoid the dreaded "pack expansion argument for non-pack
//! parameter" error.
template <template <class...> class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_indirect_quote
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG =
    typename __detail::__type_call_indirect_fn<sizeof(__type_list<_Ts...>*)>::template __call<_Fn, _Ts...>;
};

template <template <class...> class _Fn, class... _Ts>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_bind_front_quote
{
  template <class... _Us>
  using __call _CCCL_NODEBUG = _Fn<_Ts..., _Us...>;
};

//! \brief A meta-callable that always evaluates to \c _Ty.
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_always
{
  template <class...>
  using __call _CCCL_NODEBUG = _Ty;
};

//! \brief A unary meta-callable that returns its argument unmodified.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_self
{
  template <class _Ty>
  using __call _CCCL_NODEBUG = _Ty;
};

template <class _Ty>
using __type_self_t _CCCL_NODEBUG = _Ty;

//! \brief Perform a logical AND operation on a list of Boolean types.
//!
//! \note The AND operation is not short-circuiting.
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_strict_and
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __fold_and<_Ts::value...>;
};

//! \brief A pointer to a type list, often used as a function argument.
template <class... _Ts>
using __type_list_ptr = __type_list<_Ts...>*;

//! \brief Return the size of a type list.
template <class _List>
using __type_list_size = integral_constant<size_t, _List::__size>;

//! \brief Given a type list and a list of types, prepend the types to the list.
template <class _List, class... _Ts>
using __type_push_front = __type_call1<_List, __type_bind_front_quote<__type_list, _Ts...>>;

#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 9)

namespace __detail
{
template <template <class...> class _Fn, class... _Ts>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Fn<_Ts...>>*) //
  -> __type_list<_Ts...>;

template <template <class _Ty, _Ty...> class _Fn, class _Ty, _Ty... _Us>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Fn<_Ty, _Us...>>*) //
  -> __type_list<integral_constant<_Ty, _Us>...>;

template <class _Ret, class... _Args>
_CCCL_API inline auto __as_type_list_fn(__undefined<_Ret(_Args...)>*) //
  -> __type_list<_Ret, _Args...>;
} // namespace __detail

//! \brief Given a type that is can be interpreted as a type list, return its
//! type list interpretation. Types that can be interpreted as a type
//! list are of the following forms:
//!
//! \li `C<Ts...>`, for any class template `C` and types `Ts...`.
//! \li `C<T, T... Vs>`, for any class template `C`, type `T` and values `Vs...`.
//!     The resulting type is `_Fn<integral_constant<T, Vs>...>`.
//! \li `R(As...)`, for any function type `R(As...)`. The resulting type is
//!     `_Fn<R, As...>`.
template <class _List>
using __as_type_list = decltype(__detail::__as_type_list_fn(static_cast<__undefined<_List>*>(nullptr)));

#  else // ^^^ if _CCCL_CUDA_COMPILER(NVCC, <, 12, 9) / vvv if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 9) vvv

namespace __detail
{
template <class _Ret, class... _Args>
using __fn_ptr_t = _Ret (*)(_Args...);

template <class _List>
extern __undefined<_List> __as_type_list_v;

template <template <class...> class _Cy, class... _Ts>
extern __fn_ptr_t<__type_list<_Ts...>> __as_type_list_v<_Cy<_Ts...>>;

template <template <class _Ty, _Ty...> class _Fn, class _Ty, _Ty... _Us>
extern __fn_ptr_t<__type_list<std::integral_constant<_Ty, _Us>...>> __as_type_list_v<_Fn<_Ty, _Us...>>;

template <class _Ret, class... _Args>
extern __fn_ptr_t<__type_list<_Ret, _Args...>> __as_type_list_v<_Ret(_Args...)>;
} // namespace __detail

//! \brief Given a type that can be interpreted as a type list, return its
//! type list interpretation. Types that can be interpreted as a type
//! list are of the following forms:
//!
//! \li `C<Ts...>`, for any class template `C` and types `Ts...`.
//! \li `C<T, T... Vs>`, for any class template `C`, type `T` and values `Vs...`.
//!     The resulting type is `_Fn<integral_constant<T, Vs>...>`.
//! \li `R(As...)`, for any function type `R(As...)`. The resulting type is
//!     `_Fn<R, As...>`.
template <class _List>
using __as_type_list = decltype(__detail::__as_type_list_v<_List>());

#  endif // if _CCCL_CUDA_COMPILER(NVCC, >=, 12, 9)

//! \brief Given a type that can be interpreted as a type list and a
//! meta-callable, invoke the meta-callable with the types in the list.
//! \c __as_type_list is used to convert the \c _List type to a specialization
//! of \c __type_list.
//!
//! \sa __as_type_list
template <class _Fn, class _List>
using __type_apply = __type_call<__as_type_list<_List>, _Fn>;

//! \brief Given a type that can be interpreted as a type list and a class or
//! alias template, instantiate the template with the types in the list.
//! \c __as_type_list is used to convert the \c _List type to a specialization
//! of \c __type_list.
//!
//! \sa __as_type_list
template <template <class...> class _Fn, class _List>
using __type_apply_q = __type_call<__as_type_list<_List>, __type_quote<_Fn>>;

namespace __detail
{
// Only the following precise formulation works with nvcc < 12.2
template <class _Fn, class... _Ts, template <class...> class _Fn2 = _Fn::template __call, class = _Fn2<_Ts...>>
_CCCL_API inline auto __type_callable_fn(__type_list<_Fn, _Ts...>*) -> true_type;

_CCCL_API inline auto __type_callable_fn(void*) -> false_type;
} // namespace __detail

//! \brief Test whether a meta-callable is callable with a given set of
//! arguments.
//!
//! Given a meta-callable and a list of arguments, return \c true_type if
//! the meta-callable can be called with the arguments, and \c false_type
//! otherwise.
template <class _Fn, class... _Ts>
using __type_callable = decltype(__detail::__type_callable_fn(static_cast<__type_list<_Fn, _Ts...>*>(nullptr)));

//! \brief A composition of two meta-callables that will attempt to call the
//! first, and if that fails, call the second.
//!
//! Compose two meta-callables, \c _TryFn and \c _CatchFn, into a new
//! meta-callable. Given arguments \c _Ts... , \c __type_try_catch will try to
//! evaluate `__type_call<_TryFn, _Ts...>`. If this type is well-formed, it will
//! be the result of the composition. Otherwise, the result will be
//! `__type_call<_CatchFn, _Ts...>`.
template <class _TryFn, class _CatchFn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_try_catch
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __type_call<_If<__type_callable<_TryFn, _Ts...>::value, _TryFn, _CatchFn>, _Ts...>;
};

//! \brief A pair of types
template <class _First, class _Second>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_pair
{
  using __first _CCCL_NODEBUG  = _First;
  using __second _CCCL_NODEBUG = _Second;
};

//! \brief Retrieve the first of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_first _CCCL_NODEBUG = typename _Pair::__first;

//! \brief Retrieve the second of a pair of types
//! \pre \c _Pair is a specialization of \c __type_pair
template <class _Pair>
using __type_pair_second _CCCL_NODEBUG = typename _Pair::__second;

namespace __detail
{
#  if _CCCL_COMPILER(MSVC, <, 19, 38)
// A workaround for https://developercommunity.visualstudio.com/t/fatal-error-C1001:-Internal-compiler-err/10405847
struct __type_concat_fn
{
  template <class... _Lists>
  struct __trait
  {};

  template <class... _Ts, class... _Us, class... _Lists>
  struct __trait<__type_list<_Ts...>, __type_list<_Us...>, _Lists...> : __trait<__type_list<_Ts..., _Us...>, _Lists...>
  {};

  template <class... _Ts>
  struct __trait<__type_list<_Ts...>>
  {
    using type = __type_list<_Ts...>;
  };

  template <>
  struct __trait<>
  {
    using type = __type_list<>;
  };

  template <class... _Lists>
  using __call _CCCL_NODEBUG = __type<__trait<_Lists...>>;
};
#  else // ^^^ _CCCL_COMPILER(MSVC, <, 19, 38) ^^^ / vvv _CCCL_COMPILER(MSVC, >=, 19, 38) vvv
template <size_t _Count>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn
{
  using __next _CCCL_NODEBUG = __type_maybe_concat_fn<(_Count < 8 ? 0 : _Count - 8)>;

  template <class... _Ts,
            class... _As,
            class... _Bs,
            class... _Cs,
            class... _Ds,
            class... _Es,
            class... _Fs,
            class... _Gs,
            class... _Hs,
            class... _Tail>
  _CCCL_API inline static auto __fn(
    __type_list_ptr<_Ts...>, // state
    __type_list_ptr<_As...>, // 1
    __type_list_ptr<_Bs...>, // 2
    __type_list_ptr<_Cs...>, // 3
    __type_list_ptr<_Ds...>, // 4
    __type_list_ptr<_Es...>, // 5
    __type_list_ptr<_Fs...>, // 6
    __type_list_ptr<_Gs...>, // 7
    __type_list_ptr<_Hs...>, // 8
    _Tail*... __tail) // rest
    -> decltype(__next::__fn(
      __type_list_ptr<_Ts..., _As..., _Bs..., _Cs..., _Ds..., _Es..., _Fs..., _Gs..., _Hs...>{nullptr},
      __tail...,
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr},
      __type_list_ptr<>{nullptr}));
};

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_maybe_concat_fn<0>
{
  template <class... _Ts>
  _CCCL_API inline static auto __fn(__type_list_ptr<_Ts...>, ...) -> __type_list<_Ts...>;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_concat_fn
{
  template <class... _Lists>
  using __call _CCCL_NODEBUG = decltype(__type_maybe_concat_fn<sizeof...(_Lists)>::__fn(
    __type_list_ptr<>{nullptr},
    static_cast<_Lists*>(nullptr)...,
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr},
    __type_list_ptr<>{nullptr}));
};
#  endif // _CCCL_COMPILER(MSVC, >=, 19, 38)
} // namespace __detail

//! \brief Concatenate a list of type lists into a single type list.
//!
//! When passed no type lists, \c __type_concat returns an empty type list.
template <class... _Lists>
using __type_concat = __type_call<__detail::__type_concat_fn, _Lists...>;

//! \brief Given a list of type lists, concatenate all the lists into one.
//!
//! When passed an empty type list, \c __type_flatten returns an empty type list.
template <class _ListOfLists>
using __type_flatten = __type_call1<_ListOfLists, __type_quote<__type_concat>>;

namespace __detail
{
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_same_as
{
  template <class _Uy>
  using __call _CCCL_NODEBUG = bool_constant<is_same_v<_Ty, _Uy>>;
};
} // namespace __detail

//! \brief Given a type list and type, find the first occurrence of the type in
//! the list. It returns a type list containing the type and all the types after
//! it.
//!
//! If the type is not in the list, \c __type_find returns an empty type list.
template <class _List, class _Ty>
using __type_find = __type_find_if<_List, __detail::__type_same_as<_Ty>>;

namespace __detail
{
template <class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_transform_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __type_list<__type_call1<_Fn, _Ts>...>;
};
} // namespace __detail

//! \brief Given a type list and a unary meta-callable, apply the meta-callable
//! to each type in the list. It returns a new type list containing the results.
template <class _List, class _Fn>
using __type_transform = __type_call1<_List, __detail::__type_transform_fn<_Fn>>;

namespace __detail
{
template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn;

#  define _M0(_N) , class _CCCL_PP_CAT(_, _N)
#  define _M1(_N) typename _Fn::template __call <
#  define _M2(_N) , _CCCL_PP_CAT(_, _N)
#  define _M3(_N) _M2(_CCCL_PP_DEC(_N)) >

#  define _LIBCUDACXX_TYPE_FOLD_LEFT(_N)                                                                       \
    template <>                                                                                                \
    struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn<_N>                                               \
    {                                                                                                          \
      template <class _Fn, class _State _CCCL_PP_REPEAT(_N, _M0)>                                              \
      using __call _CCCL_NODEBUG = _CCCL_PP_REPEAT(_N, _M1) _State _CCCL_PP_REPEAT(_N, _M3, _N, _CCCL_PP_DEC); \
    };

_CCCL_PP_REPEAT_REVERSE(_CCCL_PP_INC(_CCCL_META_UNROLL_LIMIT), _LIBCUDACXX_TYPE_FOLD_LEFT)

template <size_t _Np>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_fn
{
  template <class _Fn, class _State _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M0), class... _Rest>
  using __call _CCCL_NODEBUG =
    __type_call<__type_fold_left_fn<_CCCL_META_UNROLL_LIMIT>,
                _Fn,
                __type_call_indirect<__type_fold_left_fn<_Np - _CCCL_META_UNROLL_LIMIT>, _Fn, _State, _Rest...> //
                  _CCCL_PP_REPEAT(_CCCL_META_UNROLL_LIMIT, _M2, 0, _CCCL_PP_INC)>;
};

template <class _Init, class _Fn>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_fold_left_select_fn
{
  template <class... _Ts>
  using __call _CCCL_NODEBUG = __type_call_indirect<__type_fold_left_fn<sizeof...(_Ts)>, _Fn, _Init, _Ts...>;
};

#  undef _LIBCUDACXX_TYPE_FOLD_LEFT

#  undef _M3
#  undef _M2
#  undef _M1
#  undef _M0
} // namespace __detail

//! \brief Fold a type list from the left with a binary meta-callable and an
//! initial state.
template <class _List, class _Init, class _Fn>
using __type_fold_left = __type_call1<_List, __detail::__type_fold_left_select_fn<_Init, _Fn>>;

namespace __detail
{
template <class _Ty>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_remove_fn
{
  template <class _Uy>
  using __call _CCCL_NODEBUG = _If<is_same_v<_Ty, _Uy>, __type_list<>, __type_list<_Uy>>;
};
} // namespace __detail

//! \brief Remove all occurrences of a type from a type list
template <class _List, class _Ty>
using __type_remove = __type_flatten<__type_transform<_List, __detail::__type_remove_fn<_Ty>>>;

//! \brief Remove all duplicate types from a type list
template <class _List>
using __type_unique = __as_type_list<__type_call<_List, __type_quote<__make_type_set>>>;

namespace __detail
{
// _State is the list of type lists being built by the __type_fold_left
template <class _State, class _List>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_cartesian_product_fn
{
  template <class _Ty>
  struct __lambda0
  {
    template <class _List2>
    using __lambda1 _CCCL_NODEBUG = __type_list<__type_push_front<_List2, _Ty>>;

    using type _CCCL_NODEBUG = __type_flatten<__type_transform<_State, __type_quote1<__lambda1>>>;
  };

  using type _CCCL_NODEBUG = __type_flatten<__type_transform<_List, __type_quote_trait1<__lambda0>>>;
};
} // namespace __detail

//! \brief Given a list of lists \c _Lists, return a new list of lists that is
//! the cartesian product.
//!
//! \par Complexity
//! `O(N * M)`, where `N` is the size of the outer list, and
//! `M` is the size of the inner lists.
template <class... _Lists>
using __type_cartesian_product =
  __type_fold_left<__type_list<_Lists...>,
                   __type_list<__type_list<>>,
                   __type_quote_trait2<__detail::__type_cartesian_product_fn>>;

//! \brief A list of compile-time values, and a meta-callable that accepts a
//! meta-callable and evaluates it with the values, each value wrapped in an
//! integral constant wrapper.
template <class _Ty, _Ty... _Values>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __type_value_list : __type_list<integral_constant<_Ty, _Values>...>
{
  using __type _CCCL_NODEBUG = _Ty;
};

namespace __detail
{
template <class _Ty, _Ty _Start, _Ty _Stride, _Ty... _Is>
_CCCL_API inline auto __type_iota_fn(integer_sequence<_Ty, _Is...>*)
  -> __type_list<integral_constant<_Ty, _Ty(_Start + (_Is * _Stride))>...>;
} // namespace __detail

//! \brief Return a \c __type_list of \c _Size integral constants starting at
//! \c _Start and incrementing by \c _Stride.
template <class _Ty, _Ty _Start, _Ty _Size, _Ty _Stride = _Ty(1)>
using __type_iota =
  decltype(__detail::__type_iota_fn<_Ty, _Start, _Stride>(static_cast<make_integer_sequence<_Ty, _Size>*>(nullptr)));

#endif // _CCCL_DOXYGEN_INVOKED

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_TYPE_LIST_OPERATIONS_H
