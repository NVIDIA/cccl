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

using Vec_f16_x1 = simd::basic_vec<__half, simd::fixed_size<1>>;

__device__ Vec_f16_x1 test_min_f16(Vec_f16_x1 a, Vec_f16_x1 b, Vec_f16_x1 c)
{
  return simd::fmin(simd::fmin(a, b), c);
}

__device__ Vec_f16_x1 test_max_f16(Vec_f16_x1 a, Vec_f16_x1 b, Vec_f16_x1 c)
{
  return simd::fmax(simd::fmax(a, b), c);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_max_f16.*}}
; SM90: {{.*VHMNMX .*!PT.*}}
; SM100: {{.*VHMNMX .*!PT.*}}
; SM103: {{.*VHMNMX .*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_min_f16.*}}
; SM90: {{.*VHMNMX .*PT.*}}
; SM100: {{.*VHMNMX .*PT.*}}
; SM103: {{.*VHMNMX .*PT.*}}

*/

#endif // _CCCL_HAS_NVFP16()
