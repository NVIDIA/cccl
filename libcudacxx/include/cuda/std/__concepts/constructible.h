//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CONCEPTS_CONSTRUCTIBLE_H
#define _CUDA_STD___CONCEPTS_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/destructible.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_HAS_CONCEPTS()

// [concept.constructible]
template <class _Tp, class... _Args>
concept constructible_from = destructible<_Tp> && is_constructible_v<_Tp, _Args...>;

// [concept.default.init]
template <class _Tp>
concept __default_initializable = requires { ::new _Tp; };

template <class _Tp>
concept default_initializable = constructible_from<_Tp> && requires { _Tp{}; } && __default_initializable<_Tp>;

// [concept.moveconstructible]
template <class _Tp>
concept move_constructible = constructible_from<_Tp, _Tp> && convertible_to<_Tp, _Tp>;

// [concept.copyconstructible]
template <class _Tp>
concept copy_constructible =
  move_constructible<_Tp> && constructible_from<_Tp, _Tp&> && convertible_to<_Tp&, _Tp>
  && constructible_from<_Tp, const _Tp&> && convertible_to<const _Tp&, _Tp> && constructible_from<_Tp, const _Tp>
  && convertible_to<const _Tp, _Tp>;

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

template <class _Tp, class... _Args>
_CCCL_CONCEPT_FRAGMENT(__constructible_from_,
                       requires()(requires(destructible<_Tp>), requires(is_constructible_v<_Tp, _Args...>)));

template <class _Tp, class... _Args>
_CCCL_CONCEPT constructible_from = _CCCL_FRAGMENT(__constructible_from_, _Tp, _Args...);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__default_initializable_, requires()((::new _Tp)));

template <class _Tp>
_CCCL_CONCEPT __default_initializable = _CCCL_FRAGMENT(__default_initializable_, _Tp);

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(_Default_initializable_,
                       requires(_Tp = _Tp{})(requires(constructible_from<_Tp>), requires(__default_initializable<_Tp>)));

template <class _Tp>
_CCCL_CONCEPT default_initializable = _CCCL_FRAGMENT(_Default_initializable_, _Tp);

// [concept.moveconstructible]
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__move_constructible_,
                       requires()(requires(constructible_from<_Tp, _Tp>), requires(convertible_to<_Tp, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT move_constructible = _CCCL_FRAGMENT(__move_constructible_, _Tp);

// [concept.copyconstructible]
template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(
  __copy_constructible_,
  requires()(
    requires(move_constructible<_Tp>),
    requires(constructible_from<_Tp, add_lvalue_reference_t<_Tp>>&& convertible_to<add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const add_lvalue_reference_t<_Tp>>&&
               convertible_to<const add_lvalue_reference_t<_Tp>, _Tp>),
    requires(constructible_from<_Tp, const _Tp>&& convertible_to<const _Tp, _Tp>)));

template <class _Tp>
_CCCL_CONCEPT copy_constructible = _CCCL_FRAGMENT(__copy_constructible_, _Tp);

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

_CCCL_END_NAMESPACE_CUDA_STD

//! The code below provides the following concepts in the ::cuda:: namespace:
//!
//! - `__list_initializable_from`
//! - `__nothrow_list_initializable_from`
//! - `__initializable_from`
//! - `__nothrow_initializable_from`
//! - `__emplaceable_from`
//! - `__nothrow_emplaceable_from`

_CCCL_BEGIN_NAMESPACE_CUDA

// constructible_from using list initialization syntax.
template <class _Tp, class... _Args>
_CCCL_CONCEPT __list_initializable_from =
  _CCCL_REQUIRES_EXPR((_Tp, variadic _Args), _Args&&... __args)(_Tp{static_cast<_Args&&>(__args)...});

template <class _Tp, class... _Args>
_CCCL_CONCEPT __nothrow_list_initializable_from =
  _CCCL_REQUIRES_EXPR((_Tp, variadic _Args), _Args&&... __args)(noexcept(_Tp{static_cast<_Args&&>(__args)...}));

//! Constructible from arguments using either direct non-list initialization or direct
//! list initialization.
template <class _Tp, class... _Args>
_CCCL_CONCEPT __initializable_from =
  ::cuda::std::constructible_from<_Tp, _Args...> || __list_initializable_from<_Tp, _Args...>;

template <class _Tp, class... _Args>
_CCCL_CONCEPT __nothrow_initializable_from =
  __initializable_from<_Tp, _Args...>
  && (::cuda::std::constructible_from<_Tp, _Args...>
        ? ::cuda::std::is_nothrow_constructible_v<_Tp, _Args...>
        : __nothrow_list_initializable_from<_Tp, _Args...>);

#if !_CCCL_COMPILER(MSVC) && !_CCCL_CUDA_COMPILER(NVCC, <, 12, 9)

//! Constructible with direct non-list initialization syntax from the result of
//! a function call expression (often useful for immovable types).
template <class _Tp, class _Fn, class... _Args>
_CCCL_CONCEPT __emplaceable_from = _CCCL_REQUIRES_EXPR((_Tp, _Fn, variadic _Args), _Fn&& __fn, _Args&&... __args)(
  _Tp(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...)));

template <class _Tp, class _Fn, class... _Args>
_CCCL_CONCEPT __nothrow_emplaceable_from =
  _CCCL_REQUIRES_EXPR((_Tp, _Fn, variadic _Args), _Fn&& __fn, _Args&&... __args)(
    noexcept(_Tp(static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...))));

#else // ^^^ !_CCCL_COMPILER(MSVC) ^^^ / vvv _CCCL_COMPILER(MSVC) vvv

//! Constructible with direct non-list initialization syntax from the result of
//! a function call expression (often useful for immovable types). MSVC cannot
//! use the above formulation because it has poor support for deferred materialization
//! of temporary object (aka, guaranteed copy elision).
template <class _Tp, class _Fn, class... _Args>
_CCCL_CONCEPT __emplaceable_from = _CCCL_REQUIRES_EXPR((_Tp, _Fn, variadic _Args), _Fn&& __fn, _Args&&... __args)(
  _Same_as(_Tp) static_cast<_Fn&&>(__fn)(static_cast<_Args&&>(__args)...));

template <class _Tp, class _Fn, class... _Args>
_CCCL_CONCEPT __nothrow_emplaceable_from =
  __emplaceable_from<_Tp, _Fn, _Args...> && ::cuda::std::__is_nothrow_callable_v<_Fn, _Args...>;

#endif // ^^^ _CCCL_COMPILER(MSVC) ^^^

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_CONSTRUCTIBLE_H
