//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_EXTENT_H
#define _CUDA_STD___TYPE_TRAITS_EXTENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CHECK_BUILTIN(array_extent)
#  define _CCCL_BUILTIN_ARRAY_EXTENT(...) __array_extent(__VA_ARGS__)
#endif // _CCCL_CHECK_BUILTIN(array_extent)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_ARRAY_EXTENT)

template <class _Tp, size_t _Ip = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : integral_constant<size_t, _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Ip)>
{};

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = _CCCL_BUILTIN_ARRAY_EXTENT(_Tp, _Ip);

#else // ^^^ _CCCL_BUILTIN_ARRAY_EXTENT ^^^ / vvv !_CCCL_BUILTIN_ARRAY_EXTENT vvv

template <class _Tp, unsigned _Ip = 0>
inline constexpr size_t extent_v = 0;
template <class _Tp>
inline constexpr size_t extent_v<_Tp[], 0> = 0;
template <class _Tp, unsigned _Ip>
inline constexpr size_t extent_v<_Tp[], _Ip> = extent_v<_Tp, _Ip - 1>;
template <class _Tp, size_t _Np>
inline constexpr size_t extent_v<_Tp[_Np], 0> = _Np;
template <class _Tp, size_t _Np, unsigned _Ip>
inline constexpr size_t extent_v<_Tp[_Np], _Ip> = extent_v<_Tp, _Ip - 1>;

template <class _Tp, unsigned _Ip = 0>
struct _CCCL_TYPE_VISIBILITY_DEFAULT extent : integral_constant<size_t, extent_v<_Tp, _Ip>>
{};

#endif // !_CCCL_BUILTIN_ARRAY_EXTENT

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_EXTENT_H
