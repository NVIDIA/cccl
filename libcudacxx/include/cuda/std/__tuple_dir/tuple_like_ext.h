//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_LIKE_EXT_H
#define _CUDA_STD___TUPLE_TUPLE_LIKE_EXT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/complex.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/complex.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
inline constexpr bool __tuple_like_ext = false;

template <class _Tp>
inline constexpr bool __tuple_like_ext<const _Tp> = __tuple_like_ext<_Tp>;
template <class _Tp>
inline constexpr bool __tuple_like_ext<volatile _Tp> = __tuple_like_ext<_Tp>;
template <class _Tp>
inline constexpr bool __tuple_like_ext<const volatile _Tp> = __tuple_like_ext<_Tp>;

template <class... _Tp>
inline constexpr bool __tuple_like_ext<tuple<_Tp...>> = true;

template <class _T1, class _T2>
inline constexpr bool __tuple_like_ext<pair<_T1, _T2>> = true;

template <class _Tp, size_t _Size>
inline constexpr bool __tuple_like_ext<array<_Tp, _Size>> = true;

template <class _Tp>
inline constexpr bool __tuple_like_ext<complex<_Tp>> = true;

template <class _Tp>
inline constexpr bool __tuple_like_ext<::cuda::complex<_Tp>> = true;

template <class... _Tp>
inline constexpr bool __tuple_like_ext<__tuple_types<_Tp...>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_LIKE_EXT_H
