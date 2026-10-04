//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/lazy.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__is_nothrow_convertible)
#  define _CCCL_IS_NOTHROW_CONVERTIBLE(...) __is_nothrow_convertible(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__is_nothrow_convertible)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_IS_NOTHROW_CONVERTIBLE)

template <class _Fm, class _To>
struct is_nothrow_convertible : bool_constant<_CCCL_IS_NOTHROW_CONVERTIBLE(_Fm, _To)>
{};

template <class _Fm, class _To>
inline constexpr bool is_nothrow_convertible_v = _CCCL_IS_NOTHROW_CONVERTIBLE(_Fm, _To);

#else // ^^^ _CCCL_IS_NOTHROW_CONVERTIBLE ^^^ / vvv !_CCCL_IS_NOTHROW_CONVERTIBLE vvv

template <class _Tp>
_CCCL_HOST_DEVICE static void __test_noexcept(_Tp) noexcept;

template <class _Fm, class _To>
_CCCL_HOST_DEVICE static bool_constant<noexcept(::cuda::std::__test_noexcept<_To>(::cuda::std::declval<_Fm>()))>
__is_nothrow_convertible_test();

template <class _Fm, class _To, bool _IsConvertible = is_convertible_v<_Fm, _To>, class = void>
inline constexpr bool __is_nothrow_convertible_impl_v = false;

template <class _Fm, class _To, class _Void>
inline constexpr bool __is_nothrow_convertible_impl_v<_Fm, _To, true, _Void> =
  decltype(::cuda::std::__is_nothrow_convertible_test<_Fm, _To>())::value;

template <class _Fm, class _To>
inline constexpr bool __is_nothrow_convertible_impl_v<_Fm, _To, true, enable_if_t<is_void_v<_Fm> && is_void_v<_To>>> =
  true;

template <class _Fm, class _To>
struct is_nothrow_convertible : bool_constant<__is_nothrow_convertible_impl_v<_Fm, _To>>
{};

template <class _Fm, class _To>
inline constexpr bool is_nothrow_convertible_v = __is_nothrow_convertible_impl_v<_Fm, _To>;

#endif // ^^^ !_CCCL_IS_NOTHROW_CONVERTIBLE ^^^

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONVERTIBLE_H
