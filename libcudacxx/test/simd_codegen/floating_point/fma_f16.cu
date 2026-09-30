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

using Vec_f16_x4 = simd::basic_vec<__half, simd::fixed_size<4>>;

__device__ Vec_f16_x4 test_fma_f16_x4(Vec_f16_x4 lhs, Vec_f16_x4 rhs, Vec_f16_x4 add)
{
  return lhs * rhs + add;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_fma_f16_x4.*}}
; SM80: {{.*HFMA2.*}}
; SM80: {{.*HFMA2.*}}
; SM90: {{.*HFMA2.*}}
; SM90: {{.*HFMA2.*}}
; SM1XX: {{.*HFMA2.*}}
; SM1XX: {{.*HFMA2.*}}

*/

#endif // _CCCL_HAS_NVFP16()
