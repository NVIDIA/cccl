//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_FULLY_BOUNDED_ARRAY_H
#define _CUDA_STD___TYPE_TRAITS_IS_FULLY_BOUNDED_ARRAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
inline constexpr bool __is_fully_bounded_array_helper_v = true;
template <class _Tp>
inline constexpr bool __is_fully_bounded_array_helper_v<_Tp[]> = false;
template <class _Tp, size_t _Np>
inline constexpr bool __is_fully_bounded_array_helper_v<_Tp[_Np]> = __is_fully_bounded_array_helper_v<_Tp>;

//! @brief Trait to test if a type is a fully bounded array, for example T[1], or T[2][1][2], but not T or T[] or T[][2]
template <class _Tp>
inline constexpr bool __is_fully_bounded_array_v = false;
template <class _Tp, size_t _Np>
inline constexpr bool __is_fully_bounded_array_v<_Tp[_Np]> = __is_fully_bounded_array_helper_v<_Tp>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_FULLY_BOUNDED_ARRAY_H
