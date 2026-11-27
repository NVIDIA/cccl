// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_ITERATOR_TAG_H
#define _CUDA_STD___ITERATOR_ITERATOR_TAG_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_complete.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/is_primary_template.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/priority_tag.h>

#if _CCCL_HOSTED()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) \
    && __has_include(<bits/stl_iterator_base_types.h>) && __has_include(<bits/stl_iterator.h>)
#    include <bits/stl_iterator_base_types.h> // for ::std::input_iterator_tag
// <bits/stl_iterator.h> only includes <bits/stl_iterator_base_types.h> itself since libstdc++ 10
#    include <bits/stl_iterator.h> // for ::__gnu_cxx::__normal_iterator
#  elif _CCCL_HOST_STD_LIB(LIBCXX) \
    && __has_include(<__iterator/iterator_traits.h>) && __has_include(<__iterator/wrap_iter.h>)
#    include <__iterator/iterator_traits.h> // for ::std::input_iterator_tag
#    include <__iterator/wrap_iter.h> // for ::std::__wrap_iter
#  elif _CCCL_HOST_STD_LIB(STL) && __has_include(<__msvc_iter_core.hpp>)
#    include <__msvc_iter_core.hpp> // for ::std::input_iterator_tag
#  else // ^^^ _CCCL_HOST_STD_LIB(STL) ^^^ / vvv other host standard library vvv
#    include <iterator> // for ::std::input_iterator_tag
#  endif // ^^^ other host standard library ^^^

#  if _CCCL_HOST_STD_LIB(LIBSTDCXX, >=, 9) && defined(_GLIBCXX_DEBUG)
#    include <debug/helper_functions.h> // for ::__gnu_debug::_Safe_iterator
#  elif _CCCL_HOST_STD_LIB(LIBSTDCXX) && defined(_GLIBCXX_DEBUG)
// <debug/helper_functions.h> only declares _Safe_iterator since libstdc++ 9, and older versions do not change anymore
namespace __gnu_debug
{
template <typename _Iterator, typename _Sequence>
class _Safe_iterator;
} // namespace __gnu_debug
#  endif // _CCCL_HOST_STD_LIB(LIBSTDCXX) && _GLIBCXX_DEBUG

#  if _CCCL_STD_VER >= 2020
#    include <cuda/std/__cccl/prologue.h>
// detect whether the used STL has contiguous_iterator_tag defined
namespace std
{
inline constexpr bool __cccl_std_contiguous_iterator_tag_exists_v =
  ::cuda::std::__is_complete_v<struct contiguous_iterator_tag>;
} // namespace std

#    include <cuda/std/__cccl/epilogue.h>
#  endif // _CCCL_STD_VER >= 2020

#endif // _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h> // NOLINT(readability-duplicate-include)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_FREESTANDING()

struct _CCCL_TYPE_VISIBILITY_DEFAULT input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT output_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT forward_iterator_tag : public input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT bidirectional_iterator_tag : public forward_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT random_access_iterator_tag : public bidirectional_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};

#else // ^^^ _CCCL_FREESTANDING() ^^^ / vvv _CCCL_HOSTED() vvv

using ::std::bidirectional_iterator_tag;
using ::std::forward_iterator_tag;
using ::std::input_iterator_tag;
using ::std::output_iterator_tag;
using ::std::random_access_iterator_tag;

#  if _CCCL_STD_VER >= 2020
struct _CCCL_TYPE_VISIBILITY_DEFAULT __contiguous_iterator_tag_backfill : public ::std::random_access_iterator_tag
{};
using contiguous_iterator_tag =
  _If<::std::__cccl_std_contiguous_iterator_tag_exists_v,
      ::std::contiguous_iterator_tag,
      __contiguous_iterator_tag_backfill>;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};
#  endif // _CCCL_STD_VER <= 2017

#endif // _CCCL_HOSTED()

template <class _Iter>
struct __iter_traits_cache
{
  using type _CCCL_NODEBUG = __select_traits<remove_cvref_t<_Iter>, remove_cvref_t<_Iter>>;
};
template <class _Iter>
using _ITER_TRAITS _CCCL_NODEBUG = typename __iter_traits_cache<_Iter>::type;

#if _CCCL_HOSTED()
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX) && defined(_GLIBCXX_DEBUG)
// The debug containers wrap the iterators of the normal ones
// The number of template parameters of _Safe_iterator depends on the version of libstdc++
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Container, class... _Args)
_CCCL_REQUIRES(
  is_same_v<_Iter, ::__gnu_debug::_Safe_iterator<::__gnu_cxx::__normal_iterator<_Ty*, _Container>, _Args...>>)
_CCCL_API auto __iter_concept_fn(
  ::__gnu_debug::_Safe_iterator<::__gnu_cxx::__normal_iterator<_Ty*, _Container>, _Args...>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#  endif // _CCCL_HOST_STD_LIB(LIBSTDCXX) && _GLIBCXX_DEBUG
#  if _CCCL_HOST_STD_LIB(LIBSTDCXX)
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Range)
_CCCL_REQUIRES(is_same_v<_Iter, ::__gnu_cxx::__normal_iterator<_Ty*, _Range>>)
_CCCL_API auto __iter_concept_fn(::__gnu_cxx::__normal_iterator<_Ty*, _Range>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#  endif // _CCCL_HOST_STD_LIB(LIBSTDCXX)
#  if _CCCL_HOST_STD_LIB(LIBCXX)
_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(is_same_v<_Iter, ::std::__wrap_iter<_Ty*>>)
_CCCL_API auto __iter_concept_fn(::std::__wrap_iter<_Ty*>, __priority_tag<3>) -> contiguous_iterator_tag;
#  elif _CCCL_HOST_STD_LIB(STL)
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_Array_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_Array_const_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_Vector_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_Vector_const_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_String_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_String_const_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_String_view_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(is_same_v<_Iter, class _Iter::_Span_iterator>)
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
#  endif // _CCCL_HOST_STD_LIB(STL)
#endif // _CCCL_HOSTED()

_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(is_same_v<_Iter, _Ty*>)
_CCCL_API auto __iter_concept_fn(_Ty*, __priority_tag<3>) -> contiguous_iterator_tag;

template <class _Iter>
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<2>) -> typename _ITER_TRAITS<_Iter>::iterator_concept;
template <class _Iter>
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<1>) -> typename _ITER_TRAITS<_Iter>::iterator_category;
template <class _Iter>
_CCCL_API auto __iter_concept_fn(_Iter, __priority_tag<0>)
  -> enable_if_t<__is_primary_cccl_template<_Iter>::value && __is_primary_std_template<_Iter>::value,
                 random_access_iterator_tag>;

template <class _Iter>
using __iter_concept_t _CCCL_NODEBUG =
  decltype(::cuda::std::__iter_concept_fn<_Iter>(::cuda::std::declval<_Iter>(), __priority_tag<3>{}));

template <class _Iter, class = void>
struct __iter_concept_cache
{};

template <class _Iter>
struct __iter_concept_cache<_Iter, void_t<__iter_concept_t<_Iter>>>
{
  using type _CCCL_NODEBUG = __iter_concept_t<_Iter>;
};

template <class _Iter>
using _ITER_CONCEPT _CCCL_NODEBUG = typename __iter_concept_cache<_Iter>::type;

template <class _Tp>
_CCCL_CONCEPT __has_member_reference = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::reference));

template <class _Tp>
_CCCL_CONCEPT __has_member_pointer = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::pointer));

template <class _Tp>
_CCCL_CONCEPT __has_member_iterator_category = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::iterator_category));

template <class _Tp>
_CCCL_CONCEPT __has_member_iterator_concept = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::iterator_concept));

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h> // NOLINT(readability-duplicate-include)

#endif // _CUDA_STD___ITERATOR_ITERATOR_TAG_H
