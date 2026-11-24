//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__simd_> // IWYU pragma: keep
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

using Vec_s32_x1 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<1>>;
using Vec_u32_x1 = simd::basic_vec<cuda::std::uint32_t, simd::fixed_size<1>>;
using Vec_s16_x2 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<2>>;
using Vec_u16_x2 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<2>>;

__device__ Vec_u32_x1 test_min_u32(Vec_u32_x1 a, Vec_u32_x1 b, Vec_u32_x1 c)
{
  return simd::min(simd::min(a, b), c);
}

__device__ Vec_s32_x1 test_min_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return simd::min(simd::min(a, b), c);
}

__device__ Vec_u32_x1 test_max_u32(Vec_u32_x1 a, Vec_u32_x1 b, Vec_u32_x1 c)
{
  return simd::max(simd::max(a, b), c);
}

__device__ Vec_s32_x1 test_max_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return simd::max(simd::max(a, b), c);
}

__device__ Vec_u16_x2 test_min_u16_x2(Vec_u16_x2 a, Vec_u16_x2 b, Vec_u16_x2 c)
{
  return simd::min(simd::min(a, b), c);
}

__device__ Vec_s16_x2 test_min_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return simd::min(simd::min(a, b), c);
}

__device__ Vec_u16_x2 test_max_u16_x2(Vec_u16_x2 a, Vec_u16_x2 b, Vec_u16_x2 c)
{
  return simd::max(simd::max(a, b), c);
}

__device__ Vec_s16_x2 test_max_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return simd::max(simd::max(a, b), c);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_s16_x2.*}}
; SM90: {{.*VIMNMX3.S16x2.*!PT.*}}
; SM100: {{.*VIMNMX3.S16x2.*!PT.*}}
; SM103: {{.*VIMNMX3.S16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_u16_x2.*}}
; SM90: {{.*VIMNMX3.U16x2.*!PT.*}}
; SM100: {{.*VIMNMX3.U16x2.*!PT.*}}
; SM103: {{.*VIMNMX3.U16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_s16_x2.*}}
; SM90: {{.*VIMNMX3.S16x2.*PT.*}}
; SM100: {{.*VIMNMX3.S16x2.*PT.*}}
; SM103: {{.*VIMNMX3.S16x2.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_u16_x2.*}}
; SM90: {{.*VIMNMX3.U16x2.*PT.*}}
; SM100: {{.*VIMNMX3.U16x2.*PT.*}}
; SM103: {{.*VIMNMX3.U16x2.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_s32.*}}
; SM90: {{.*VIMNMX3 .*!PT.*}}
; SM100: {{.*VIMNMX3 .*!PT.*}}
; SM103: {{.*VIMNMX3 .*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_u32.*}}
; SM90: {{.*VIMNMX3.U32.*!PT.*}}
; SM100: {{.*VIMNMX3.U32.*!PT.*}}
; SM103: {{.*VIMNMX3.U32.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_s32.*}}
; SM90: {{.*VIMNMX3 .*PT.*}}
; SM100: {{.*VIMNMX3 .*PT.*}}
; SM103: {{.*VIMNMX3 .*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_u32.*}}
; SM90: {{.*VIMNMX3.U32.*PT.*}}
; SM100: {{.*VIMNMX3.U32.*PT.*}}
; SM103: {{.*VIMNMX3.U32.*PT.*}}

*/
