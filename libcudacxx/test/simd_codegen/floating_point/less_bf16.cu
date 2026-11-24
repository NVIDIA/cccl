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

using Vec_bf16_x4  = simd::basic_vec<__nv_bfloat16, simd::fixed_size<4>>;
using Mask_bf16_x4 = Vec_bf16_x4::mask_type;

__device__ Mask_bf16_x4 test_less_bf16_x4(Vec_bf16_x4 lhs, Vec_bf16_x4 rhs)
{
  return lhs < rhs;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_less_bf16_x4.*}}
; SM80: {{.*FSETP\.LT.*}}
; SM80: {{.*FSETP\.LT.*}}
; SM80: {{.*FSETP\.LT.*}}
; SM80: {{.*FSETP\.LT.*}}
; SM90: {{.*HSETP2.*BF16.*}}
; SM90: {{.*HSETP2.*BF16.*}}
; SM1XX: {{.*HSETP2.*BF16.*}}
; SM1XX: {{.*HSETP2.*BF16.*}}

*/

#endif // _CCCL_HAS_NVBF16()
