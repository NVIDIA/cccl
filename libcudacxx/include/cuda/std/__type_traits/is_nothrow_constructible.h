//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
#define _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_constructible : public integral_constant<bool, _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...)>
{};

template <class _Tp, class... _Args>
inline constexpr bool is_nothrow_constructible_v = _CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE(_Tp, _Args...);

#else

template <bool, class _Tp, class... _Args>
inline constexpr bool __cccl_is_nothrow_constructible = false;

template <class _Tp, class... _Args>
inline constexpr bool __cccl_is_nothrow_constructible<true, _Tp, _Args...> =
  noexcept(_Tp(::cuda::std::declval<_Args>()...));

template <class _Tp, class _Arg>
inline constexpr bool __cccl_is_nothrow_constructible<true, _Tp, _Arg> =
  noexcept(static_cast<_Tp>(::cuda::std::declval<_Arg>()));

template <class _Tp, size_t _Ns>
inline constexpr bool __cccl_is_nothrow_constructible<true, _Tp[_Ns]> = __cccl_is_nothrow_constructible<true, _Tp>;

template <class _Tp, class... _Args>
inline constexpr bool is_nothrow_constructible_v =
  __cccl_is_nothrow_constructible<is_constructible_v<_Tp, _Args...>, _Tp, _Args...>;

template <class _Tp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_constructible : bool_constant<is_nothrow_constructible_v<_Tp, _Args...>>
{};

#endif // defined(_CCCL_BUILTIN_IS_NOTHROW_CONSTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_NOTHROW_CONSTRUCTIBLE_FALLBACK)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_NOTHROW_CONSTRUCTIBLE_H
