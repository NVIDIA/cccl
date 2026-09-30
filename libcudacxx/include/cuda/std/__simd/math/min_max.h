//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_MIN_MAX_H
#define _CUDA_STD___SIMD_MATH_MIN_MAX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/fdim.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_BINARY_GENERATOR(fdim, fdim);
_CCCL_SIMD_MATH_BINARY_GENERATOR(fmax, fmax);
_CCCL_SIMD_MATH_BINARY_GENERATOR(fmin, fmin);

_CCCL_SIMD_MATH_BINARY_FUNCTION(fdim, fdim, )
_CCCL_SIMD_MATH_BINARY_FUNCTION(fmax, fmax, constexpr)
_CCCL_SIMD_MATH_BINARY_FUNCTION(fmin, fmin, constexpr)

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::fdim;
using simd::fmax;
using simd::fmin;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_MIN_MAX_H
