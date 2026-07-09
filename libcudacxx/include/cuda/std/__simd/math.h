//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_H
#define _CUDA_STD___SIMD_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/math/abs.h>
#include <cuda/std/__simd/math/classification.h>
#include <cuda/std/__simd/math/common.h>
#include <cuda/std/__simd/math/comparison.h>
#include <cuda/std/__simd/math/exponential.h>
#include <cuda/std/__simd/math/fma_lerp.h>
#include <cuda/std/__simd/math/manipulation.h>
#include <cuda/std/__simd/math/min_max.h>
#include <cuda/std/__simd/math/modulo.h>
#include <cuda/std/__simd/math/rounding.h>
#include <cuda/std/__simd/math/trigonometric.h>

#endif // _CUDA_STD___SIMD_MATH_H
