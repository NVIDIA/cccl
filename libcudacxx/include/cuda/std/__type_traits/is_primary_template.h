//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/void_t.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <iterator>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_COMPILER(MSVC)

template <class _Iter, class = void>
struct __is_primary_cccl_template : false_type
{};

template <class _Iter>
struct __is_primary_cccl_template<_Iter, void_t<typename iterator_traits<_Iter>::__cccl_primary_template>>
    : public is_same<iterator_traits<_Iter>, typename iterator_traits<_Iter>::__cccl_primary_template>
{};

#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv

template <class _Traits>
using __test_for_primary_template = enable_if_t<_IsSame<_Traits, typename _Traits::__cccl_primary_template>::value>;
template <class _Iter>
using __is_primary_cccl_template = _IsValidExpansion<__test_for_primary_template, iterator_traits<_Iter>>;

#endif // !_CCCL_COMPILER(MSVC)

#if _CCCL_COMPILER(NVRTC)

// No ::std::traits with NVRTC
template <class _Iter>
struct __is_primary_std_template : true_type
{};

template <class _Iter, class _OtherTraits>
using __select_traits = conditional_t<__is_primary_cccl_template<_Iter>::value, _OtherTraits, iterator_traits<_Iter>>;

#else // ^^^ _CCCL_COMPILER(NVRTC) ^^^ / vvv !_CCCL_COMPILER(NVRTC) vvv

// We also need to respect what the user is defining to std::iterator_traits
#  if defined(__GLIBCXX__)
// libstdc++ uses `__is_base_of`
template <class _Iter>
using __test_for_primary_std_template =
  enable_if_t<_CCCL_TRAIT(is_base_of, ::std::__iterator_traits<_Iter, void>, ::std::iterator_traits<_Iter>)>;
template <class _Iter>
using __is_primary_std_template = _IsValidExpansion<__test_for_primary_std_template, _Iter>;
#  elif defined(_LIBCPP_VERSION)
// libc++ uses the same mechanism than we do with __primary_template
template <class _Traits>
using __test_for_primary_std_template = enable_if_t<_IsSame<_Traits, typename _Traits::__primary_template>::value>;
template <class _Iter>
using __is_primary_std_template = _IsValidExpansion<__test_for_primary_template, ::std::iterator_traits<_Iter>>;
#  elif defined(_MSVC_STL_VERSION) || defined(_IS_WRS)
// MSVC uses the same mechanism than we do with _From_primary
template <class _Iter, class = void>
struct __is_primary_std_template : false_type
{};

template <class _Iter>
struct __is_primary_std_template<_Iter, void_t<typename ::std::iterator_traits<_Iter>::_From_primary>>
    : public is_same<::std::iterator_traits<_Iter>, typename ::std::iterator_traits<_Iter>::_From_primary>
{};
#  endif // _MSVC_STL_VERSION || _IS_WRS

// C++17 has issues with e.g void* and other pointers.
// C++20 fails subsumption if we use the indirection
#  if !defined(_CCCL_NO_CONCEPTS)
template <class _Iter, class _OtherTraits>
using __select_traits =
  conditional_t<__is_primary_std_template<_Iter>::value,
                conditional_t<__is_primary_cccl_template<_Iter>::value, _OtherTraits, iterator_traits<_Iter>>,
                ::std::iterator_traits<_Iter>>;

#  else // ^^^ !_CCCL_NO_CONCEPTS ^^^ / vvv _CCCL_NO_CONCEPTS vvv

template <class _Iter, class _OtherTraits, bool = _CCCL_TRAIT(is_pointer, _Iter)>
struct __select_traits_impl
{
  using type =
    conditional_t<__is_primary_std_template<_Iter>::value,
                  conditional_t<__is_primary_cccl_template<_Iter>::value, _OtherTraits, iterator_traits<_Iter>>,
                  ::std::iterator_traits<_Iter>>;
};

// Pointers are treated specially. Also guards against void* issues
template <class _Iter, class _OtherTraits>
struct __select_traits_impl<_Iter, _OtherTraits, true>
{
  using type = iterator_traits<_Iter>;
};

template <class _Iter, class _OtherTraits>
using __select_traits = typename __select_traits_impl<_Iter, _OtherTraits>::type;

#  endif // _CCCL_NO_CONCEPTS

#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
