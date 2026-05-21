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

using Vec_s8_x4 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;
using Vec_u8_x4 = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;

__device__ Vec_u8_x4 test_min_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return simd::min(lhs, rhs);
}

__device__ Vec_s8_x4 test_min_s8_x4(Vec_s8_x4 lhs, Vec_s8_x4 rhs)
{
  return simd::min(lhs, rhs);
}

__device__ Vec_u8_x4 test_max_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return simd::max(lhs, rhs);
}

__device__ Vec_s8_x4 test_max_s8_x4(Vec_s8_x4 lhs, Vec_s8_x4 rhs)
{
  return simd::max(lhs, rhs);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_s8_x4.*}}
; SM120f: {{.*VIMNMX.S8x4.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_u8_x4.*}}
; SM120f: {{.*VIMNMX.U8x4.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_s8_x4.*}}
; SM120f: {{.*VIMNMX.S8x4.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_u8_x4.*}}
; SM120f: {{.*VIMNMX.U8x4.*PT.*}}

*/
