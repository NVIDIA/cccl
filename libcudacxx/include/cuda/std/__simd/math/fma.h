//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_FMA_H
#define _CUDA_STD___SIMD_MATH_FMA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__simd/math/common.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

_CCCL_TEMPLATE(typename _Vp0, typename _Vp1, typename _Vp2)
_CCCL_REQUIRES(__is_simd_math_v<__simd_math_result_t<_Vp0, _Vp1, _Vp2>, _Vp0, _Vp1, _Vp2>)
[[nodiscard]] _CCCL_HOST_DEVICE_API auto fma(const _Vp0& __x, const _Vp1& __y, const _Vp2& __z) noexcept
{
  using __result_t = __simd_math_result_t<_Vp0, _Vp1, _Vp2>;
  const __result_t __x_vec{__x};
  const __result_t __y_vec{__y};
  const __result_t __z_vec{__z};
  return __simd_fma_impl(__x_vec, __y_vec, __z_vec); // ADL
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using simd::fma;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_FMA_H
