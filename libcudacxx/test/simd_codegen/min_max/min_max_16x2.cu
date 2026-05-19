//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__simd/specializations/min_max_optimization.h>
#include <cuda/std/__simd_> // IWYU pragma: keep

namespace simd = cuda::std::simd;

using Vec_s16_x2 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<2>>;
using Vec_u16_x2 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<2>>;

__device__ Vec_u16_x2 test_min_u16_x2(Vec_u16_x2 lhs, Vec_u16_x2 rhs)
{
  return simd::min(lhs, rhs);
}

__device__ Vec_s16_x2 test_min_s16_x2(Vec_s16_x2 lhs, Vec_s16_x2 rhs)
{
  return simd::min(lhs, rhs);
}

__device__ Vec_u16_x2 test_max_u16_x2(Vec_u16_x2 lhs, Vec_u16_x2 rhs)
{
  return simd::max(lhs, rhs);
}

__device__ Vec_s16_x2 test_max_s16_x2(Vec_s16_x2 lhs, Vec_s16_x2 rhs)
{
  return simd::max(lhs, rhs);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_s16_x2.*}}
; SM90: {{.*VIMNMX.S16x2.*!PT.*}}
; SM1XX: {{.*VIMNMX.S16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_u16_x2.*}}
; SM90: {{.*VIMNMX.U16x2.*!PT.*}}
; SM1XX: {{.*VIMNMX.U16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_s16_x2.*}}
; SM90: {{.*VIMNMX.S16x2.*PT.*}}
; SM1XX: {{.*VIMNMX.S16x2.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_u16_x2.*}}
; SM90: {{.*VIMNMX.U16x2.*PT.*}}
; SM1XX: {{.*VIMNMX.U16x2.*PT.*}}

*/
