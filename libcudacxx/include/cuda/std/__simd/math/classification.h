//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_CLASSIFICATION_H
#define _CUDA_STD___SIMD_MATH_CLASSIFICATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/fpclassify.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/isinf.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__cmath/isnormal.h>
#include <cuda/std/__cmath/signbit.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_UNARY_GENERATOR(fpclassify);
_CCCL_SIMD_MATH_UNARY_GENERATOR(isfinite);
_CCCL_SIMD_MATH_UNARY_GENERATOR(isinf);
_CCCL_SIMD_MATH_UNARY_GENERATOR(isnan);
_CCCL_SIMD_MATH_UNARY_GENERATOR(isnormal);
_CCCL_SIMD_MATH_UNARY_GENERATOR(signbit);

#define _CCCL_SIMD_MATH_MASK_FUNCTION(_NAME, _CONSTEXPR)                                    \
  _CCCL_TEMPLATE(typename _Vp, typename _Result = typename __deduced_vec_t<_Vp>::mask_type) \
  _CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)                                           \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp& __x) noexcept                 \
  {                                                                                         \
    return _Result{__simd_##_NAME##_generator<_Result, _Vp>{__x}};                          \
  }

//----------------------------------------------------------------------------------------------------------------------

_CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(fpclassify, int, constexpr)

_CCCL_SIMD_MATH_MASK_FUNCTION(isfinite, constexpr)
_CCCL_SIMD_MATH_MASK_FUNCTION(isinf, constexpr)
_CCCL_SIMD_MATH_MASK_FUNCTION(isnan, constexpr)
_CCCL_SIMD_MATH_MASK_FUNCTION(isnormal, constexpr)
_CCCL_SIMD_MATH_MASK_FUNCTION(signbit, constexpr)

#undef _CCCL_SIMD_MATH_MASK_FUNCTION

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::fpclassify;
using simd::isfinite;
using simd::isinf;
using simd::isnan;
using simd::isnormal;
using simd::signbit;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_CLASSIFICATION_H
