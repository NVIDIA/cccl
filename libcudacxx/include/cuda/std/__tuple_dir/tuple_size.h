//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_SIZE_H
#define _CUDA_STD___TUPLE_TUPLE_SIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_volatile.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size;

template <class _Tp, class...>
using __enable_if_tuple_size_imp = _Tp;

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const _Tp,
                                      enable_if_t<!is_volatile_v<_Tp>>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<volatile _Tp,
                                      enable_if_t<!is_const_v<_Tp>>,
                                      integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
tuple_size<__enable_if_tuple_size_imp<const volatile _Tp, integral_constant<size_t, sizeof(tuple_size<_Tp>)>>>
    : integral_constant<size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
inline constexpr size_t tuple_size_v = tuple_size<_Tp>::value;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_STD

template <class _Tp>
struct tuple_size;

#if _CCCL_FREESTANDING()

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<const _Tp,
                                          ::cuda::std::enable_if_t<!::cuda::std::is_volatile_v<_Tp>>,
                                          ::cuda::std::integral_constant<::cuda::std::size_t, sizeof(tuple_size<_Tp>)>>>
    : ::cuda::std::integral_constant<::cuda::std::size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<volatile _Tp,
                                          ::cuda::std::enable_if_t<!::cuda::std::is_const_v<_Tp>>,
                                          ::cuda::std::integral_constant<::cuda::std::size_t, sizeof(tuple_size<_Tp>)>>>
    : ::cuda::std::integral_constant<::cuda::std::size_t, tuple_size<_Tp>::value>
{};

template <class _Tp>
struct tuple_size<
  ::cuda::std::__enable_if_tuple_size_imp<const volatile _Tp,
                                          ::cuda::std::integral_constant<::cuda::std::size_t, sizeof(tuple_size<_Tp>)>>>
    : ::cuda::std::integral_constant<::cuda::std::size_t, tuple_size<_Tp>::value>
{};
#endif // _CCCL_FREESTANDING()

_CCCL_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_SIZE_H
