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

#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/tie.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_like.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class _Up>
struct __tuple_cat_type;

template <class... _Ttypes, class... _Utypes>
struct __tuple_cat_type<tuple<_Ttypes...>, __tuple_types<_Utypes...>>
{
  using type _CCCL_NODEBUG_ALIAS = tuple<_Ttypes..., _Utypes...>;
};

template <class _ResultTuple, bool _Is_Tuple0TupleLike, class... _Tuples>
struct __tuple_cat_return_1
{};

template <class... _Types, class _Tuple0>
struct __tuple_cat_return_1<tuple<_Types...>, true, _Tuple0>
{
  using type _CCCL_NODEBUG_ALIAS =
    typename __tuple_cat_type<tuple<_Types...>, __make_tuple_types_t<remove_cvref_t<_Tuple0>>>::type;
};

template <class... _Types, class _Tuple0, class _Tuple1, class... _Tuples>
struct __tuple_cat_return_1<tuple<_Types...>, true, _Tuple0, _Tuple1, _Tuples...>
    : public __tuple_cat_return_1<
        typename __tuple_cat_type<tuple<_Types...>, __make_tuple_types_t<remove_cvref_t<_Tuple0>>>::type,
        __tuple_like_impl<remove_reference_t<_Tuple1>>,
        _Tuple1,
        _Tuples...>
{};

template <class... _Tuples>
struct __tuple_cat_return;

template <class _Tuple0, class... _Tuples>
struct __tuple_cat_return<_Tuple0, _Tuples...>
    : public __tuple_cat_return_1<tuple<>, __tuple_like_impl<remove_reference_t<_Tuple0>>, _Tuple0, _Tuples...>
{};

template <>
struct __tuple_cat_return<>
{
  using type _CCCL_NODEBUG_ALIAS = tuple<>;
};

_CCCL_API constexpr tuple<> tuple_cat()
{
  return tuple<>();
}

template <class _Rp, class _Indices, class _Tuple0, class... _Tuples>
struct __tuple_cat_return_ref_imp;

template <class... _Types, size_t... _I0, class _Tuple0>
struct __tuple_cat_return_ref_imp<tuple<_Types...>, __tuple_indices<_I0...>, _Tuple0>
{
  using _T0 _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tuple0>;
  using type                    = tuple<_Types..., __copy_cvref_t<_Tuple0, tuple_element_t<_I0, _T0>>&&...>;
};

template <class... _Types, size_t... _I0, class _Tuple0, class _Tuple1, class... _Tuples>
struct __tuple_cat_return_ref_imp<tuple<_Types...>, __tuple_indices<_I0...>, _Tuple0, _Tuple1, _Tuples...>
    : public __tuple_cat_return_ref_imp<
        tuple<_Types..., __copy_cvref_t<_Tuple0, tuple_element_t<_I0, remove_reference_t<_Tuple0>>>&&...>,
        __make_tuple_indices_t<tuple_size<remove_reference_t<_Tuple1>>::value>,
        _Tuple1,
        _Tuples...>
{};

template <class _Tuple0, class... _Tuples>
struct __tuple_cat_return_ref
    : public __tuple_cat_return_ref_imp<tuple<>,
                                        __make_tuple_indices_t<tuple_size<remove_reference_t<_Tuple0>>::value>,
                                        _Tuple0,
                                        _Tuples...>
{};

template <class _Types, class _I0, class _J0>
struct __tuple_cat;

template <class... _Types, size_t... _I0, size_t... _J0>
struct __tuple_cat<tuple<_Types...>, __tuple_indices<_I0...>, __tuple_indices<_J0...>>
{
  template <class _Tuple0>
  _CCCL_API constexpr typename __tuple_cat_return_ref<tuple<_Types...>&&, _Tuple0&&>::type
  operator()([[maybe_unused]] tuple<_Types...> __t, _Tuple0&& __t0)
  {
    return ::cuda::std::forward_as_tuple(::cuda::std::forward<_Types>(::cuda::std::get<_I0>(__t))...,
                                         ::cuda::std::get<_J0>(::cuda::std::forward<_Tuple0>(__t0))...);
  }

  template <class _Tuple0, class _Tuple1, class... _Tuples>
  _CCCL_API constexpr typename __tuple_cat_return_ref<tuple<_Types...>&&, _Tuple0&&, _Tuple1&&, _Tuples&&...>::type
  operator()([[maybe_unused]] tuple<_Types...> __t, _Tuple0&& __t0, _Tuple1&& __t1, _Tuples&&... __tpls)
  {
    using _T0 _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tuple0>;
    using _T1 _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tuple1>;
    return __tuple_cat<tuple<_Types..., __copy_cvref_t<_Tuple0, tuple_element_t<_J0, _T0>>&&...>,
                       __make_tuple_indices_t<sizeof...(_Types) + tuple_size<_T0>::value>,
                       __make_tuple_indices_t<tuple_size<_T1>::value>>()(
      ::cuda::std::forward_as_tuple(::cuda::std::forward<_Types>(::cuda::std::get<_I0>(__t))...,
                                    ::cuda::std::get<_J0>(::cuda::std::forward<_Tuple0>(__t0))...),
      ::cuda::std::forward<_Tuple1>(__t1),
      ::cuda::std::forward<_Tuples>(__tpls)...);
  }
};

template <class _Tuple0, class... _Tuples>
_CCCL_API constexpr typename __tuple_cat_return<_Tuple0, _Tuples...>::type tuple_cat(_Tuple0&& __t0, _Tuples&&... __tpls)
{
  using _T0 _CCCL_NODEBUG_ALIAS = remove_reference_t<_Tuple0>;
  return __tuple_cat<tuple<>, __tuple_indices<>, __make_tuple_indices_t<tuple_size<_T0>::value>>()(
    tuple<>(), ::cuda::std::forward<_Tuple0>(__t0), ::cuda::std::forward<_Tuples>(__tpls)...);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_CAT_H
