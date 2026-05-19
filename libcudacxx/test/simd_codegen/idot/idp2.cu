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

using Vec_s8_x4  = simd::basic_vec<cuda::std::int8_t, simd::fixed_size<4>>;
using Vec_u8_x4  = simd::basic_vec<cuda::std::uint8_t, simd::fixed_size<4>>;
using Vec_s16_x4 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<4>>;
using Vec_u16_x4 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<4>>;

__device__ cuda::std::int32_t test_idot_s16_s8(Vec_s16_x4 lhs, Vec_s8_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::int32_t test_idot_s8_s16(Vec_s8_x4 lhs, Vec_s16_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::int32_t test_idot_s16_u8(Vec_s16_x4 lhs, Vec_u8_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::int32_t test_idot_u8_s16(Vec_u8_x4 lhs, Vec_s16_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::int32_t test_idot_u16_s8(Vec_u16_x4 lhs, Vec_s8_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::int32_t test_idot_s8_u16(Vec_s8_x4 lhs, Vec_u16_x4 rhs, cuda::std::int32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::uint32_t test_idot_u16_u8(Vec_u16_x4 lhs, Vec_u8_x4 rhs, cuda::std::uint32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

__device__ cuda::std::uint32_t test_idot_u8_u16(Vec_u8_x4 lhs, Vec_u16_x4 rhs, cuda::std::uint32_t acc)
{
  return cuda::simd::idot(lhs, rhs, acc);
}

/*

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_s16_s8.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_s8_s16.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_s16_u8.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_u8_s16.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_u16_s8.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_s8_u16.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_u16_u8.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

; SMXX-DAG: {{[[:space:]]*}}Function : {{.*test_idot_u8_u16.*}}
; SMXX-DAG: {{.*IDP\.2A.*LO.*}}
; SMXX-DAG: {{.*IDP\.2A.*HI.*}}

*/
