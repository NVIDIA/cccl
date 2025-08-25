//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FWD_COMPLEX_H
#define _CUDA___FWD_COMPLEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT complex;

// __is_cuda_complex_v

template <class _Tp>
inline constexpr bool __is_cuda_complex_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_complex_v<const _Tp> = __is_cuda_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_complex_v<volatile _Tp> = __is_cuda_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_complex_v<const volatile _Tp> = __is_cuda_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_complex_v<complex<_Tp>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FWD_COMPLEX_H
