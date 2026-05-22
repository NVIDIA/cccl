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

using Vec_u16_x2 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<2>>;

__device__ Vec_u16_x2 test_operator_plus_u16_x2(Vec_u16_x2 lhs, Vec_u16_x2 rhs)
{
  return lhs + rhs;
}

__device__ Vec_u16_x2 test_operator_minus_u16_x2(Vec_u16_x2 lhs, Vec_u16_x2 rhs)
{
  return lhs - rhs;
}

__device__ Vec_u16_x2 test_operator_post_decrement_u16_x2(Vec_u16_x2 in)
{
  (void) in--;
  return in;
}

__device__ Vec_u16_x2 test_operator_post_increment_u16_x2(Vec_u16_x2 in)
{
  (void) in++;
  return in;
}

__device__ Vec_u16_x2 test_operator_unary_minus_u16_x2(Vec_u16_x2 in)
{
  return -in;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_unary_minus_u16_x2.*}}
; SM90: {{.*VIADD.*}}
; SM1XX: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_post_increment_u16_x2.*}}
; SM90: {{.*VIADD.*}}
; SM1XX: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_post_decrement_u16_x2.*}}
; SM90: {{.*VIADD.*}}
; SM1XX: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_minus_u16_x2.*}}
; SM90: {{.*VIADD.*}}
; SM1XX: {{.*VIADD.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_plus_u16_x2.*}}
; SM90: {{.*VIADD.*}}
; SM1XX: {{.*VIADD.*}}

*/
