//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/simd> // IWYU pragma: keep
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

using Vec_s32_x1 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<1>>;
using Vec_s16_x2 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<2>>;
using Vec_s8_x4  = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;

__device__ Vec_s8_x4 test_max_relu_s8_x4(Vec_s8_x4 a, Vec_s8_x4 b)
{
  return cuda::simd::max_relu(a, b);
}

__device__ Vec_s8_x4 test_min_relu_s8_x4(Vec_s8_x4 a, Vec_s8_x4 b)
{
  return cuda::simd::min_relu(a, b);
}

__device__ Vec_s8_x4 test_max_relu_s8_x4_3way(Vec_s8_x4 a, Vec_s8_x4 b, Vec_s8_x4 c)
{
  return cuda::simd::max_relu(a, b, c);
}

__device__ Vec_s8_x4 test_min_relu_s8_x4_3way(Vec_s8_x4 a, Vec_s8_x4 b, Vec_s8_x4 c)
{
  return cuda::simd::min_relu(a, b, c);
}

__device__ Vec_s16_x2 test_max_relu_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b)
{
  return cuda::simd::max_relu(a, b);
}

__device__ Vec_s16_x2 test_min_relu_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b)
{
  return cuda::simd::min_relu(a, b);
}

__device__ Vec_s16_x2 test_max_relu_s16_x2_3way(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::max_relu(a, b, c);
}

__device__ Vec_s16_x2 test_min_relu_s16_x2_3way(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::min_relu(a, b, c);
}

__device__ Vec_s32_x1 test_max_relu_s32(Vec_s32_x1 a, Vec_s32_x1 b)
{
  return cuda::simd::max_relu(a, b);
}

__device__ Vec_s32_x1 test_min_relu_s32(Vec_s32_x1 a, Vec_s32_x1 b)
{
  return cuda::simd::min_relu(a, b);
}

__device__ Vec_s32_x1 test_max_relu_s32_3way(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::max_relu(a, b, c);
}

__device__ Vec_s32_x1 test_min_relu_s32_3way(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::min_relu(a, b, c);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s32_3way.*}}
; SM90: {{.*VIMNMX3\.RELU.*PT.*}}
; SM100: {{.*VIMNMX3\.RELU.*PT.*}}
; SM103: {{.*VIMNMX3\.RELU.*PT.*}}
; SM107: {{.*VIMNMX.*RELU.*PT.*}}
; SM107: {{.*VIMNMX.*RELU.*PT.*}}
; SM120: {{.*VIMNMX.*RELU.*PT.*}}
; SM120: {{.*VIMNMX.*RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s32_3way.*}}
; SM90: {{.*VIMNMX3\.RELU.*!PT.*}}
; SM100: {{.*VIMNMX3\.RELU.*!PT.*}}
; SM103: {{.*VIMNMX3\.RELU.*!PT.*}}
; SM107: {{.*VIMNMX.*RELU.*!PT.*}}
; SM107: {{.*VIMNMX.*RELU.*!PT.*}}
; SM120: {{.*VIMNMX.*RELU.*!PT.*}}
; SM120: {{.*VIMNMX.*RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s32.*}}
; SM90: {{.*VIMNMX.*RELU.*PT.*}}
; SM100: {{.*VIMNMX.*RELU.*PT.*}}
; SM103: {{.*VIMNMX.*RELU.*PT.*}}
; SM107: {{.*VIMNMX.*RELU.*PT.*}}
; SM120: {{.*VIMNMX.*RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s32.*}}
; SM90: {{.*VIMNMX.*RELU.*!PT.*}}
; SM100: {{.*VIMNMX.*RELU.*!PT.*}}
; SM103: {{.*VIMNMX.*RELU.*!PT.*}}
; SM107: {{.*VIMNMX.*RELU.*!PT.*}}
; SM120: {{.*VIMNMX.*RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s16_x2_3way.*}}
; SM90: {{.*VIMNMX3\.S16x2\.RELU.*PT.*}}
; SM100: {{.*VIMNMX3\.S16x2\.RELU.*PT.*}}
; SM103: {{.*VIMNMX3\.S16x2\.RELU.*PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s16_x2_3way.*}}
; SM90: {{.*VIMNMX3\.S16x2\.RELU.*!PT.*}}
; SM100: {{.*VIMNMX3\.S16x2\.RELU.*!PT.*}}
; SM103: {{.*VIMNMX3\.S16x2\.RELU.*!PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s16_x2.*}}
; SM90: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM100: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM103: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s16_x2.*}}
; SM90: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM100: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM103: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s8_x4_3way.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s8_x4_3way.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_relu_s8_x4.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_relu_s8_x4.*}}
; SM107f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}
; SM120f: {{.*VIMNMX\.S8x4\.RELU.*!PT.*}}

*/
