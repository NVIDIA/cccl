//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_TRIGONOMETRIC_H
#define _CUDA_STD___SIMD_MATH_TRIGONOMETRIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/hyperbolic_functions.h>
#include <cuda/std/__cmath/inverse_hyperbolic_functions.h>
#include <cuda/std/__cmath/inverse_trigonometric_functions.h>
#include <cuda/std/__cmath/trigonometric_functions.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_UNARY_FUNCTION(acos, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(asin, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(atan, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(cos, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(sin, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(tan, )

_CCCL_SIMD_MATH_UNARY_FUNCTION(acosh, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(asinh, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(atanh, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(cosh, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(sinh, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(tanh, )

_CCCL_SIMD_MATH_BINARY_GENERATOR(atan2, atan2);

_CCCL_SIMD_MATH_BINARY_FUNCTION(atan2, atan2, )

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::acos;
using simd::acosh;
using simd::asin;
using simd::asinh;
using simd::atan;
using simd::atan2;
using simd::atanh;
using simd::cos;
using simd::cosh;
using simd::sin;
using simd::sinh;
using simd::tan;
using simd::tanh;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_TRIGONOMETRIC_H
