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
#include <cuda/std/array>

#if _CCCL_HAS_NVBF16()

#  include <cuda_bf16.h>

namespace simd = cuda::std::simd;

using Vec_bf16_4 = simd::basic_vec<__nv_bfloat16, simd::fixed_size<4>>;

extern "C" __global__ void
test_operator_multiplies_bf16_4(const __nv_bfloat16* lhs, const __nv_bfloat16* rhs, __nv_bfloat16* out)
{
  const cuda::std::array<__nv_bfloat16, 4> lhs_values{lhs[0], lhs[1], lhs[2], lhs[3]};
  const cuda::std::array<__nv_bfloat16, 4> rhs_values{rhs[0], rhs[1], rhs[2], rhs[3]};

  const Vec_bf16_4 lhs_vec(lhs_values);
  const Vec_bf16_4 rhs_vec(rhs_values);
  const Vec_bf16_4 result = lhs_vec * rhs_vec;

  out[0] = result[0];
  out[1] = result[1];
  out[2] = result[2];
  out[3] = result[3];
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : test_operator_multiplies_bf16_4
; SM80: {{.*HFMA2.*BF16.*}}
; SM80: {{.*HFMA2.*BF16.*}}
; SM90: {{.*HFMA2.*BF16.*}}
; SM90: {{.*HMUL2.*BF16.*}}
; SM100: {{.*HFMA2.*BF16.*}}
; SM100: {{.*HMUL2.*BF16.*}}
; SM120: {{.*HFMA2.*BF16.*}}
; SM120: {{.*HMUL2.*BF16.*}}

*/

#endif // _CCCL_HAS_NVBF16()
