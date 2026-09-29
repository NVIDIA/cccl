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

using vec_t = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<5>>;

extern "C" __device__ cuda::std::int32_t test_dot_tail(vec_t lhs, vec_t rhs, cuda::std::int32_t init)
{
  return cuda::simd::dot(lhs, rhs, init);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_dot_tail
; SMXX: {{.*IDP\.4A\.S8\.S8.*}}
; SMXX: {{.*IDP\.4A\.S8\.S8.*}}

*/
