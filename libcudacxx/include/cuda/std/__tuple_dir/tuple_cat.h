//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_CAT_H
#define _CUDA_STD___TUPLE_TUPLE_CAT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/tie.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __tuple_cat_return_impl(__tuple_types<_Types...>) noexcept
  -> __tuple_types<_Types...>
{
  return {};
}

template <class... _Types1, class... _Types2, class... _TupleTypes>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto
__tuple_cat_return_impl(__tuple_types<_Types1...>, __tuple_types<_Types2...>, _TupleTypes... __tail) noexcept
{
  return ::cuda::std::__tuple_cat_return_impl(__tuple_types<_Types1..., _Types2...>{}, __tail...);
}

template <class... _Types>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL tuple<_Types...> __tuple_cat_return_type(__tuple_types<_Types...>) noexcept;

template <class... _Tuples>
using __tuple_cat_return_t = decltype(::cuda::std::__tuple_cat_return_type(
  ::cuda::std::__tuple_cat_return_impl(__make_tuple_types_t<remove_cvref_t<_Tuples>>{}...)));

_CCCL_EXEC_CHECK_DISABLE
template <class _Tuple, size_t... _Indices>
[[nodiscard]] _CCCL_API constexpr auto __tuple_cat_impl(__tuple_indices<_Indices...>, _Tuple&& __tuple) noexcept
{
  using ::cuda::std::get;
  return ::cuda::std::forward_as_tuple(get<_Indices>(::cuda::std::forward<_Tuple>(__tuple))...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tuple1, class _Tuple2, size_t... _Indices1, size_t... _Indices2, class... _Tuples>
[[nodiscard]] _CCCL_API constexpr auto __tuple_cat_impl(
  __tuple_indices<_Indices1...>,
  __tuple_indices<_Indices2...>,
  _Tuple1&& __tuple1,
  _Tuple2&& __tuple2,
  _Tuples&&... __tuples)
{
  using ::cuda::std::get;
  if constexpr (sizeof...(_Tuples) != 0)
  {
    using _TupleSize0 = __make_tuple_indices_t<sizeof...(_Indices1) + sizeof...(_Indices2)>;
    using _TupleSize1 = __make_tuple_indices_t<tuple_size<remove_reference_t<__type_index_c<0, _Tuples...>>>::value>;
    return ::cuda::std::__tuple_cat_impl(
      _TupleSize0{},
      _TupleSize1{},
      ::cuda::std::forward_as_tuple(get<_Indices1>(::cuda::std::forward<_Tuple1>(__tuple1))...,
                                    get<_Indices2>(::cuda::std::forward<_Tuple2>(__tuple2))...),
      ::cuda::std::forward<_Tuples>(__tuples)...);
  }
  else
  {
    return ::cuda::std::forward_as_tuple(get<_Indices1>(::cuda::std::forward<_Tuple1>(__tuple1))...,
                                         get<_Indices2>(::cuda::std::forward<_Tuple2>(__tuple2))...);
  }
}

template <class... _Tuples>
_CCCL_CONCEPT __all_tuple_like = (__tuple_like<_Tuples> && ...);

[[nodiscard]] _CCCL_API constexpr tuple<> tuple_cat()
{
  return tuple<>();
}

_CCCL_TEMPLATE(class... _Tuples)
_CCCL_REQUIRES(__all_tuple_like<_Tuples...>)
[[nodiscard]] _CCCL_API constexpr __tuple_cat_return_t<_Tuples...> tuple_cat(_Tuples&&... __tuples)
{
  if constexpr (sizeof...(_Tuples) <= 2)
  {
    return ::cuda::std::__tuple_cat_impl(__make_tuple_indices_t<tuple_size<remove_reference_t<_Tuples>>::value>{}...,
                                         ::cuda::std::forward<_Tuples>(__tuples)...);
  }
  else
  {
    using _TupleSize0 = __make_tuple_indices_t<tuple_size<remove_reference_t<__type_index_c<0, _Tuples...>>>::value>;
    using _TupleSize1 = __make_tuple_indices_t<tuple_size<remove_reference_t<__type_index_c<1, _Tuples...>>>::value>;
    return ::cuda::std::__tuple_cat_impl(_TupleSize0{}, _TupleSize1{}, ::cuda::std::forward<_Tuples>(__tuples)...);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CAT_H
