//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_SWAPPABLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_SWAPPABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/referenceable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_move_assignable.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// We need to detect whether there is already a free function swap that would end up being ambiguous.
// This can happen when a type pulls in both namespace std and namespace cuda::std via ADL.
// In that case we are always safe to just not do anything because that type must be host only.
// However, we must be careful to ensure that we still create the overload if there is just a hidden friend swap
namespace __detect_hidden_friend_swap
{
// This will intentionally create an ambiguity with std::swap if that is find-able by ADL. But it will not interfere
// with hidden friend swap
template <class _Tp>
_CCCL_HOST_DEVICE void swap(_Tp&, _Tp&); // NOLINT(performance-noexcept-swap)

struct __hidden_friend_swap_found
{};

template <class _Tp>
_CCCL_API inline auto __swap(_Tp& __lhs, _Tp& __rhs) -> decltype(swap(__lhs, __rhs));
_CCCL_API inline auto __swap(...) -> __hidden_friend_swap_found;

// Note: Cannot be a concept, or the constraints of swap are cyclic
template <class _Tp>
inline constexpr bool __has_hidden_friend_swap =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __val)(_Same_as(void) __detect_hidden_friend_swap::__swap(__val, __val));
} // namespace __detect_hidden_friend_swap

namespace __detect_adl_swap
{
template <class _Tp>
void swap(_Tp&, _Tp&) = delete;

struct __no_adl_swap_found
{};

template <class _Tp>
_CCCL_API inline auto __swap(_Tp& __lhs, _Tp& __rhs) -> decltype(swap(__lhs, __rhs));
_CCCL_API inline auto __swap(...) -> __no_adl_swap_found;

// Note: Cannot be a concept, or the constraints of swap are cyclic
template <class _Tp>
inline constexpr bool __has_no_adl_swap =
  _CCCL_REQUIRES_EXPR((_Tp), _Tp& __val)(_Same_as(__no_adl_swap_found) __detect_adl_swap::__swap(__val, __val));
} // namespace __detect_adl_swap

template <class _Tp>
struct __is_swappable;
template <class _Tp>
struct __is_nothrow_swappable;

template <class _Tp>
using __swap_result_t _CCCL_NODEBUG =
  enable_if_t<(__detect_adl_swap::__has_no_adl_swap<_Tp> || __detect_hidden_friend_swap::__has_hidden_friend_swap<_Tp>)
              && is_move_constructible_v<_Tp> && is_move_assignable_v<_Tp>>;

// we use type_identity_t<_Tp> as second parameter, to avoid ambiguity with std::swap, which will thus be preferred by
// overload resolution (which is ok since std::swap is only considered when explicitly called, or found by ADL for types
// from std::)
template <class _Tp>
_CCCL_API constexpr __swap_result_t<_Tp> swap(_Tp& __x, type_identity_t<_Tp>& __y) noexcept(
  is_nothrow_move_constructible_v<_Tp> && is_nothrow_move_assignable_v<_Tp>);

template <class _Tp, size_t _Np>
_CCCL_API constexpr enable_if_t<__detect_adl_swap::__has_no_adl_swap<_Tp (&)[_Np]> && __is_swappable<_Tp>::value>
  swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) noexcept(__is_nothrow_swappable<_Tp>::value);

namespace __detail
{
// ALL generic swap overloads MUST already have a declaration available at this point.

template <class _Tp, class _Up = _Tp>
_CCCL_CONCEPT __swappable_with = _CCCL_REQUIRES_EXPR((_Tp, _Up), _Tp __x, _Up __y)(
  swap(static_cast<_Tp&&>(__x), static_cast<_Up&&>(__y)), //
  swap(static_cast<_Up&&>(__y), static_cast<_Tp&&>(__x)));

// Not replaced with a concept because GCC cannot mangle the noexcept function declaration
template <class _Tp, class _Up = _Tp, bool _Swappable = __swappable_with<_Tp, _Up>>
inline constexpr bool __nothrow_swappable_with =
  noexcept(swap(::cuda::std::declval<_Tp>(), ::cuda::std::declval<_Up>()))
  && noexcept(swap(::cuda::std::declval<_Up>(), ::cuda::std::declval<_Tp>()));

template <class _Tp, class _Up>
inline constexpr bool __nothrow_swappable_with<_Tp, _Up, false> = false;
} // namespace __detail

template <class _Tp>
struct __is_swappable : public integral_constant<bool, __detail::__swappable_with<_Tp&>>
{};

template <class _Tp>
struct __is_nothrow_swappable : public integral_constant<bool, __detail::__nothrow_swappable_with<_Tp&>>
{};

template <class _Tp, class _Up>
inline constexpr bool is_swappable_with_v = __detail::__swappable_with<_Tp, _Up>;

template <class _Tp, bool = __referenceable<_Tp>>
inline constexpr bool is_swappable_v = false;

template <class _Tp>
inline constexpr bool is_swappable_v<_Tp, true> =
  is_swappable_with_v<add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<_Tp>>;

template <class _Tp, class _Up>
inline constexpr bool is_nothrow_swappable_with_v = __detail::__nothrow_swappable_with<_Tp, _Up>;

template <class _Tp, bool = __referenceable<_Tp>>
inline constexpr bool is_nothrow_swappable_v = false;

template <class _Tp>
inline constexpr bool is_nothrow_swappable_v<_Tp, true> =
  is_nothrow_swappable_with_v<add_lvalue_reference_t<_Tp>, add_lvalue_reference_t<_Tp>>;

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_swappable_with : public bool_constant<is_swappable_with_v<_Tp, _Up>>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_swappable : public bool_constant<is_swappable_v<_Tp>>
{};

template <class _Tp, class _Up>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_swappable_with : public bool_constant<is_nothrow_swappable_with_v<_Tp, _Up>>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_swappable : public bool_constant<is_nothrow_swappable_v<_Tp>>
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_SWAPPABLE_H
