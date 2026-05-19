//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_COMPARISON_H
#define _CUDA_STD___SIMD_MATH_COMPARISON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/traits.h>
#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_SIMD_MATH_BINARY_GENERATOR(isgreater, isgreater);
_CCCL_SIMD_MATH_BINARY_GENERATOR(isgreaterequal, isgreaterequal);
_CCCL_SIMD_MATH_BINARY_GENERATOR(isless, isless);
_CCCL_SIMD_MATH_BINARY_GENERATOR(islessequal, islessequal);
_CCCL_SIMD_MATH_BINARY_GENERATOR(islessgreater, islessgreater);
_CCCL_SIMD_MATH_BINARY_GENERATOR(isunordered, isunordered);

#define _CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(_NAME, _CONSTEXPR)                               \
  _CCCL_TEMPLATE(typename _Vp0,                                                               \
                 typename _Vp1,                                                               \
                 typename _Vec    = __simd_math_result_t<_Vp0, _Vp1>,                         \
                 typename _Result = typename _Vec::mask_type)                                 \
  _CCCL_REQUIRES(__is_simd_math_v<_Vec, _Vp0, _Vp1>)                                          \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp0& __x, const _Vp1& __y) noexcept \
  {                                                                                           \
    const _Vec __x_vec = __x;                                                                 \
    const _Vec __y_vec = __y;                                                                 \
    return _Result{__simd_##_NAME##_generator<_Result, _Vec, _Vec>{__x_vec, __y_vec}};        \
  }

_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(isgreater, )
_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(isgreaterequal, )
_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(isless, )
_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(islessequal, )
_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(islessgreater, )
_CCCL_SIMD_MATH_BINARY_MASK_FUNCTION(isunordered, )

#undef _CCCL_SIMD_MATH_BINARY_MASK_FUNCTION

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::isgreater;
using simd::isgreaterequal;
using simd::isless;
using simd::islessequal;
using simd::islessgreater;
using simd::isunordered;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_COMPARISON_H
