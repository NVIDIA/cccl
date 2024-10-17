//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
#define _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/add_cv.h>
#include <cuda/std/__type_traits/add_volatile.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element;

template <size_t _Ip, class... _Tp>
using __tuple_element_t _LIBCUDACXX_NODEBUG_TYPE = typename tuple_element<_Ip, _Tp...>::type;

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const _Tp>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE typename add_const<__tuple_element_t<_Ip, _Tp>>::type type;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, volatile _Tp>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE typename add_volatile<__tuple_element_t<_Ip, _Tp>>::type type;
};

template <size_t _Ip, class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, const volatile _Tp>
{
  typedef _LIBCUDACXX_NODEBUG_TYPE typename add_cv<__tuple_element_t<_Ip, _Tp>>::type type;
};

template <size_t _Ip, class... _Types>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, __tuple_types<_Types...>>
{
  static_assert(_Ip < sizeof...(_Types), "tuple_element index out of range");
  typedef _LIBCUDACXX_NODEBUG_TYPE __type_index_c<_Ip, _Types...> type;
};

#if _CCCL_STD_VER > 2011
template <size_t _Ip, class... _Tp>
using tuple_element_t _LIBCUDACXX_NODEBUG_TYPE = typename tuple_element<_Ip, _Tp...>::type;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
