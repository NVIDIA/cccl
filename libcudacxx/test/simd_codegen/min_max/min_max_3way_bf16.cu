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

#if _CCCL_HAS_NVBF16()

#  include <cuda_bf16.h>

namespace simd = cuda::std::simd;

using Vec_bf16_x1 = simd::basic_vec<__nv_bfloat16, simd::fixed_size<1>>;

__device__ Vec_bf16_x1 test_min_bf16(Vec_bf16_x1 a, Vec_bf16_x1 b, Vec_bf16_x1 c)
{
  return simd::fmin(simd::fmin(a, b), c);
}

__device__ Vec_bf16_x1 test_max_bf16(Vec_bf16_x1 a, Vec_bf16_x1 b, Vec_bf16_x1 c)
{
  return simd::fmax(simd::fmax(a, b), c);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_bf16.*}}
; SM90: {{.*VHMNMX.BF16_V2.*!PT.*}}
; SM100: {{.*VHMNMX.BF16_V2.*!PT.*}}
; SM103: {{.*VHMNMX.BF16_V2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_bf16.*}}
; SM90: {{.*VHMNMX.BF16_V2.*PT.*}}
; SM100: {{.*VHMNMX.BF16_V2.*PT.*}}
; SM103: {{.*VHMNMX.BF16_V2.*PT.*}}

*/

#endif // _CCCL_HAS_NVBF16()
