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

using Vec_s8_x4 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;

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

/*

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
