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

using Vec_s16_x2 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<2>>;
using Vec_s8_x4  = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;
using Vec_u16_x2 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<2>>;
using Vec_u8_x4  = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;

__device__ Vec_u16_x2 test_saturating_add_u16_x2(Vec_u16_x2 lhs, Vec_u16_x2 rhs)
{
  return cuda::simd::saturating_add(lhs, rhs);
}

__device__ Vec_s16_x2 test_saturating_add_s16_x2(Vec_s16_x2 lhs, Vec_s16_x2 rhs)
{
  return cuda::simd::saturating_add(lhs, rhs);
}

__device__ Vec_u8_x4 test_saturating_add_u8_x4(Vec_u8_x4 lhs, Vec_u8_x4 rhs)
{
  return cuda::simd::saturating_add(lhs, rhs);
}

__device__ Vec_s8_x4 test_saturating_add_s8_x4(Vec_s8_x4 lhs, Vec_s8_x4 rhs)
{
  return cuda::simd::saturating_add(lhs, rhs);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_saturating_add_s8_x4.*}}
; SM120f: {{.*VIADD\.S8x4\.ISAT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_saturating_add_u8_x4.*}}
; SM120f: {{.*VIADD\.U8x4\.ISAT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_saturating_add_s16_x2.*}}
; SM120f: {{.*VIADD\.S16x2\.ISAT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_saturating_add_u16_x2.*}}
; SM120f: {{.*VIADD\.16x2\.ISAT.*}}

*/
