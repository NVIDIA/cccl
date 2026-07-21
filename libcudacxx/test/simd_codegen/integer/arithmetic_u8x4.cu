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

namespace simd = cuda::std::simd;

using Vec_u8_x4 = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;

__device__ Vec_u8_x4 test_operator_plus_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return lhs + rhs;
}

__device__ Vec_u8_x4 test_operator_minus_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return lhs - rhs;
}

__device__ Vec_u8_x4 test_operator_post_decrement_u8_x4(Vec_u8_x4 in)
{
  (void) in--;
  return in;
}

__device__ Vec_u8_x4 test_operator_post_increment_u8_x4(Vec_u8_x4 in)
{
  (void) in++;
  return in;
}

__device__ Vec_u8_x4 test_operator_unary_minus_u8_x4(Vec_u8_x4 in)
{
  return -in;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_unary_minus_u8_x4.*}}
; SM120f: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_post_increment_u8_x4.*}}
; SM120f: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_post_decrement_u8_x4.*}}
; SM120f: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_minus_u8_x4.*}}
; SM120f: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_plus_u8_x4.*}}
; SM120f: {{.*VIADD.*}}

*/
