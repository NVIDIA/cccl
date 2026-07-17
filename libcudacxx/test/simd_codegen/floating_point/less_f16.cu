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

#if _CCCL_HAS_NVFP16()

#  include <cuda_fp16.h>

namespace simd = cuda::std::simd;

using Vec_f16_x4  = simd::basic_vec<__half, simd::fixed_size<4>>;
using Mask_f16_x4 = Vec_f16_x4::mask_type;

__device__ Mask_f16_x4 test_less_f16_x4(Vec_f16_x4 lhs, Vec_f16_x4 rhs)
{
  return lhs < rhs;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_less_f16_x4.*}}
; SM80: {{.*HSETP2.*}}
; SM80: {{.*HSETP2.*}}
; SM90: {{.*HSETP2.*}}
; SM90: {{.*HSETP2.*}}
; SM1XX: {{.*HSETP2.*}}
; SM1XX: {{.*HSETP2.*}}

*/

#endif // _CCCL_HAS_NVFP16()
