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

using Vec_f32_x4 = simd::basic_vec<float, simd::fixed_size<4>>;

__device__ Vec_f32_x4 test_fma_f32_x4(Vec_f32_x4 lhs, Vec_f32_x4 rhs, Vec_f32_x4 add)
{
  return simd::fma(lhs, rhs, add);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_fma_f32_x4.*}}
; SM100: {{.*FFMA2.*}}
; SM100: {{.*FFMA2.*}}
; SM103: {{.*FFMA2.*}}
; SM103: {{.*FFMA2.*}}

*/
