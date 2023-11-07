//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FWD_GET_H
#define _LIBCUDACXX___FWD_GET_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#include "../__fwd/array.h"
#include "../__fwd/pair.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/tuple_element.h"
#include "../cstddef"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class... _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, tuple<_Tp...>>& get(
    tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, tuple<_Tp...>>& get(
    const tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, tuple<_Tp...>>&& get(
    tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class... _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, tuple<_Tp...>>&& get(
    const tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, pair<_T1, _T2>>& get(
    pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, pair<_T1, _T2>>& get(
    const pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __tuple_element_t<_Ip, pair<_T1, _T2>>&& get(
    pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const __tuple_element_t<_Ip, pair<_T1, _T2>>&& get(
    const pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Tp& get(array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _Tp& get(const array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 _Tp&& get(array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 const _Tp&& get(const array<_Tp, _Size>&&) noexcept;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FWD_GET_H
