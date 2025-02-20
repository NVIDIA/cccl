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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/void_t.h>

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

template <class _Iter, class _OtherTraits>
using __select_traits = conditional_t<__is_primary_cccl_template<_Iter>::value, _OtherTraits, iterator_traits<_Iter>>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_PRIMARY_TEMPLATE_H
