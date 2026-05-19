//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/simd> // IWYU pragma: keep

namespace simd = cuda::std::simd;

using Vec_s8_x4 = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;
using Vec_u8_x4 = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;

__device__ Vec_u8_x4 test_abs_diff_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return cuda::simd::abs_diff(lhs, rhs);
}

__device__ Vec_u8_x4 test_abs_diff_s8_x4(Vec_s8_x4 lhs, Vec_s8_x4 rhs)
{
  return cuda::simd::abs_diff(lhs, rhs);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_abs_diff_s8_x4.*}}
; SM80: {{.*VABSDIFF4 .*}}
; SM90: {{.*VABSDIFF4 .*}}
; SM100: {{.*VABSDIFF4 .*}}
; SM120: {{.*VIMNMX\.S8x4.*}}
; SM120: {{.*VIMNMX\.S8x4.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_abs_diff_u8_x4.*}}
; SM80: {{.*VABSDIFF4\.U8.*}}
; SM90: {{.*VABSDIFF4\.U8.*}}
; SM100: {{.*VABSDIFF4\.U8.*}}
; SM120: {{.*VIMNMX\.U8x4.*}}
; SM120: {{.*VIMNMX\.U8x4.*}}

*/
