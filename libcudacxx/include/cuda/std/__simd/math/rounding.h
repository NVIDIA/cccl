//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_ROUNDING_H
#define _CUDA_STD___SIMD_MATH_ROUNDING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_UNARY_FUNCTION(ceil, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(floor, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(nearbyint, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(rint, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(round, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(trunc, )

_CCCL_SIMD_MATH_UNARY_GENERATOR(lrint);
_CCCL_SIMD_MATH_UNARY_GENERATOR(llrint);
_CCCL_SIMD_MATH_UNARY_GENERATOR(lround);
_CCCL_SIMD_MATH_UNARY_GENERATOR(llround);

_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(lrint, long, )
_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(llrint, long long, )
_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(lround, long, )
_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(llround, long long, )

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::ceil;
using simd::floor;
using simd::llrint;
using simd::llround;
using simd::lrint;
using simd::lround;
using simd::nearbyint;
using simd::rint;
using simd::round;
using simd::trunc;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_ROUNDING_H
