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
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstddef>

#if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Set, class... _Ty>
_LIBCUDACXX_INLINE_VAR constexpr bool __type_set_contains =
  _CUDA_VSTD::__fold_and<_CCCL_TRAIT(_CUDA_VSTD::is_base_of, __type_identity<_Ty>, _Set)...>;

namespace __set
{
template <class... _Ts>
struct __tupl
{
  static constexpr size_t __size() noexcept
  {
    return 0;
  }
};

template <class _Ty, class... _Ts>
struct __tupl<_Ty, _Ts...>
    : __type_identity<_Ty>
    , __tupl<_Ts...>
{
  static constexpr size_t __size() noexcept
  {
    return sizeof...(_Ts) + 1;
  }
};

template <class _Ty, class... _Elements>
using __insert = _If<__type_set_contains<__tupl<_Elements...>, _Ty>, __tupl<_Elements...>, __tupl<_Ty, _Elements...>>;

struct __bulk_insert
{
  template <class... _Ts>
  static auto __call(__tupl<_Ts...>*) -> __tupl<_Ts...>;

  template <class _Ap, class... _Us, class... _Ts, class _SetInsert = __bulk_insert>
  static auto __call(__tupl<_Ts...>*)
    -> decltype(_SetInsert::template __call<_Us...>(static_cast<__insert<_Ap, _Ts...>*>(nullptr)));
};
} // namespace __set

template <class _ExpectedSet, class... _Ts>
_LIBCUDACXX_INLINE_VAR constexpr bool __type_set_eq = //
  (sizeof...(_Ts) == _ExpectedSet::__size()) && __type_set_contains<_ExpectedSet, _Ts...>;

template <class... _Ts>
using __type_set = __set::__tupl<_Ts...>;

template <class _Set, class... _Ts>
using __type_set_insert = decltype(__set::__bulk_insert::__call<_Ts...>(static_cast<_Set*>(nullptr)));

template <class... _Ts>
using __make_type_set = __type_set_insert<__type_set<>, _Ts...>;

template <class _Ty, class... _Ts>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_included_in = __fold_or<_CCCL_TRAIT(is_same, _Ty, _Ts)...>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _CCCL_STD_VER >= 2014

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_SET_H
