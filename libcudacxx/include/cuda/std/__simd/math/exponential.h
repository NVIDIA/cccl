//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_EXPONENTIAL_H
#define _CUDA_STD___SIMD_MATH_EXPONENTIAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/error_functions.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/gamma.h>
#include <cuda/std/__cmath/hypot.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/roots.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_UNARY_FUNCTION(exp, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(exp2, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(expm1, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(log, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(log10, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(log1p, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(log2, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(cbrt, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(sqrt, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(erf, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(erfc, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(lgamma, )
_CCCL_SIMD_MATH_UNARY_FUNCTION(tgamma, )

_CCCL_SIMD_MATH_BINARY_GENERATOR(pow, pow);
_CCCL_SIMD_MATH_BINARY_GENERATOR(hypot, hypot_two_args);
_CCCL_SIMD_MATH_TERNARY_GENERATOR(hypot, hypot_three_args);

_CCCL_SIMD_MATH_BINARY_FUNCTION(pow, pow, )
_CCCL_SIMD_MATH_BINARY_FUNCTION(hypot, hypot_two_args, )
_CCCL_SIMD_MATH_TERNARY_FUNCTION(hypot, hypot_three_args, )

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::cbrt;
using simd::erf;
using simd::erfc;
using simd::exp;
using simd::exp2;
using simd::expm1;
using simd::hypot;
using simd::lgamma;
using simd::log;
using simd::log10;
using simd::log1p;
using simd::log2;
using simd::pow;
using simd::sqrt;
using simd::tgamma;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_EXPONENTIAL_H
