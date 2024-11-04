//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Set, class... _Ty>
struct __type_set_contains : __fold_and<_CCCL_TRAIT(is_base_of, __type_identity<_Ty>, _Set)...>
{};

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class _Set, class... _Ty>
_CCCL_INLINE_VAR constexpr bool __type_set_contains_v = __fold_and_v<is_base_of_v<__type_identity<_Ty>, _Set>...>;
#endif // _CCCL_NO_VARIABLE_TEMPLATES

namespace __set
{
template <class... _Ts>
struct __tupl;

template <>
struct __tupl<>
{
  template <class _Ty>
  using __maybe_insert _LIBCUDACXX_NODEBUG_TYPE = __tupl<_Ty>;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __size() noexcept
  {
    return 0;
  }
};

template <class _Ty, class... _Ts>
struct __tupl<_Ty, _Ts...>
    : __type_identity<_Ty>
    , __tupl<_Ts...>
{
  template <class _Uy>
  using __maybe_insert _LIBCUDACXX_NODEBUG_TYPE =
    _If<_CCCL_TRAIT(__type_set_contains, __tupl, _Uy), __tupl, __tupl<_Uy, _Ty, _Ts...>>;

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __size() noexcept
  {
    return sizeof...(_Ts) + 1;
  }
};

template <bool _Empty>
struct __bulk_insert
{
  template <class _Set, class...>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Set;
};

template <>
struct __bulk_insert<false>
{
  template <class _Set, class _Ty, class... _Us>
  using __call _LIBCUDACXX_NODEBUG_TYPE =
    typename __bulk_insert<sizeof...(_Us) == 0>::template __call<typename _Set::template __maybe_insert<_Ty>, _Us...>;
};
} // namespace __set

// When comparing sets for equality, use conjunction<> to short-circuit the set
// comparison if the sizes are different.
template <class _ExpectedSet, class... _Ts>
using __type_set_eq =
  conjunction<bool_constant<sizeof...(_Ts) == _ExpectedSet::__size()>, __type_set_contains<_ExpectedSet, _Ts...>>;

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class _ExpectedSet, class... _Ts>
_CCCL_INLINE_VAR constexpr bool __type_set_eq_v = __type_set_eq<_ExpectedSet, _Ts...>::value;
#endif // _CCCL_NO_VARIABLE_TEMPLATES

template <class... _Ts>
using __type_set = __set::__tupl<_Ts...>;

template <class _Set, class... _Ts>
using __type_set_insert = typename __set::__bulk_insert<sizeof...(_Ts) == 0>::template __call<_Set, _Ts...>;

template <class... _Ts>
using __make_type_set = __type_set_insert<__type_set<>, _Ts...>;

template <class _Ty, class... _Ts>
struct __is_included_in : __fold_or<_CCCL_TRAIT(is_same, _Ty, _Ts)...>
{};

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class _Ty, class... _Ts>
_CCCL_INLINE_VAR constexpr bool __is_included_in_v = __fold_or_v<is_same_v<_Ty, _Ts>...>;
#endif // _CCCL_NO_VARIABLE_TEMPLATES

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H
