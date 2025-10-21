//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_COMPLEX_H
#define _CUDA_STD___FWD_COMPLEX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

// std:: forward declarations

#if _CCCL_HAS_HOST_STD_LIB()
_CCCL_BEGIN_NAMESPACE_STD

template <class>
class complex;

_CCCL_END_NAMESPACE_STD
#endif // _CCCL_HAS_HOST_STD_LIB()

// cuda::std:: forward declarations

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT complex;

// __is_std_complex_v

template <class _Tp>
inline constexpr bool __is_std_complex_v = false;
template <class _Tp>
inline constexpr bool __is_std_complex_v<const _Tp> = __is_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_std_complex_v<volatile _Tp> = __is_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_std_complex_v<const volatile _Tp> = __is_std_complex_v<_Tp>;
#if !_CCCL_COMPILER(NVRTC)
template <class _Tp>
inline constexpr bool __is_std_complex_v<::std::complex<_Tp>> = true;
#endif // !_CCCL_COMPILER(NVRTC)

// __is_cuda_std_complex_v

template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v = false;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<const _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<volatile _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<const volatile _Tp> = __is_cuda_std_complex_v<_Tp>;
template <class _Tp>
inline constexpr bool __is_cuda_std_complex_v<complex<_Tp>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_COMPLEX_H
