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

__device__ Vec_f32_x4 test_operator_decrement_f32_x4(Vec_f32_x4 vec)
{
  --vec;
  return vec;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_operator_decrement_f32_x4.*}}
; SM100: {{.*FADD2.*}}
; SM100: {{.*FADD2.*}}

*/
