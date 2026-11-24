//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/simd>
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

using lhs_vec_t = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<5>>;
using rhs_vec_t = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<5>>;

extern "C" __device__ cuda::std::int32_t test_dot_tail(lhs_vec_t lhs, rhs_vec_t rhs, cuda::std::int32_t init)
{
  return cuda::simd::dot(lhs, rhs, init);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_dot_tail
; SMXX: {{.*IDP\.2A\.LO\.S16\.S8.*}}
; SMXX: {{.*IDP\.2A\.HI\.S16\.S8.*}}
; SMXX: {{.*IDP\.2A\.LO\.S16\.S8.*}}

*/
