//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/boolean_testable.h>
#include <cuda/std/__type_traits/make_const_lvalue_ref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4389) // '==': signed/unsigned mismatch
_CCCL_DIAG_SUPPRESS_MSVC(4018) // '<': signed/unsigned mismatch

template <class _Tp, class _Up, class = void>
inline constexpr bool __is_cpp17_equality_comparable_v = false;

template <class _Tp, class _Up>
inline constexpr bool __is_cpp17_equality_comparable_v<
  _Tp,
  _Up,
  void_t<decltype(::cuda::std::declval<__make_const_lvalue_ref<_Tp>>()
                  == ::cuda::std::declval<__make_const_lvalue_ref<_Up>>())>> =
  __boolean_testable<decltype(::cuda::std::declval<__make_const_lvalue_ref<_Tp>>()
                              == ::cuda::std::declval<__make_const_lvalue_ref<_Up>>())>;

template <class _Tp, class _Up, bool = __is_cpp17_equality_comparable_v<_Tp, _Up>>
inline constexpr bool __is_cpp17_nothrow_equality_comparable_v = false;

template <class _Tp, class _Up>
inline constexpr bool __is_cpp17_nothrow_equality_comparable_v<_Tp, _Up, true> = noexcept(
  ::cuda::std::declval<__make_const_lvalue_ref<_Tp>>() == ::cuda::std::declval<__make_const_lvalue_ref<_Up>>());

template <class _Tp, class _Up, class = void>
inline constexpr bool __is_cpp17_less_than_comparable_v = false;

template <class _Tp, class _Up>
inline constexpr bool __is_cpp17_less_than_comparable_v<
  _Tp,
  _Up,
  void_t<decltype(::cuda::std::declval<__make_const_lvalue_ref<_Tp>>()
                  < ::cuda::std::declval<__make_const_lvalue_ref<_Up>>())>> =
  __boolean_testable<decltype(::cuda::std::declval<__make_const_lvalue_ref<_Tp>>()
                              < ::cuda::std::declval<__make_const_lvalue_ref<_Up>>())>;

template <class _Tp, class _Up, bool = __is_cpp17_less_than_comparable_v<_Tp, _Up>>
inline constexpr bool __is_cpp17_nothrow_less_than_comparable_v = false;

template <class _Tp, class _Up>
inline constexpr bool __is_cpp17_nothrow_less_than_comparable_v<_Tp, _Up, true> =
  noexcept(::cuda::std::declval<__make_const_lvalue_ref<_Tp>>() < ::cuda::std::declval<__make_const_lvalue_ref<_Up>>());

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_COMPARABLE_H
