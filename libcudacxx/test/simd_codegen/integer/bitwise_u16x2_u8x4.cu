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

namespace simd = cuda::std::simd;

using cuda::std::uint16_t;
using cuda::std::uint8_t;

using Vec_u16_x2 = simd::basic_vec<uint16_t, simd::fixed_size<2>>;
using Vec_u8_x4  = simd::basic_vec<uint8_t, simd::fixed_size<4>>;

__device__ void test_bitwise_and_u16_x2(Vec_u16_x2& out, Vec_u16_x2& lhs, Vec_u16_x2& rhs)
{
  out = lhs & rhs;
}

__device__ void test_bitwise_or_u16_x2(Vec_u16_x2& out, Vec_u16_x2& lhs, Vec_u16_x2& rhs)
{
  out = lhs | rhs;
}

__device__ void test_bitwise_xor_u16_x2(Vec_u16_x2& out, Vec_u16_x2& lhs, Vec_u16_x2& rhs)
{
  out = lhs ^ rhs;
}

__device__ void test_bitwise_not_u16_x2(Vec_u16_x2& out, Vec_u16_x2& in)
{
  out = ~in;
}

__device__ void test_bitwise_and_u8_x4(Vec_u8_x4& out, Vec_u8_x4& lhs, Vec_u8_x4& rhs)
{
  out = lhs & rhs;
}

__device__ void test_bitwise_or_u8_x4(Vec_u8_x4& out, Vec_u8_x4& lhs, Vec_u8_x4& rhs)
{
  out = lhs | rhs;
}

__device__ void test_bitwise_xor_u8_x4(Vec_u8_x4& out, Vec_u8_x4& lhs, Vec_u8_x4& rhs)
{
  out = lhs ^ rhs;
}

__device__ void test_bitwise_not_u8_x4(Vec_u8_x4& out, Vec_u8_x4& in)
{
  out = ~in;
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_not_u8_x4.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_xor_u8_x4.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_or_u8_x4.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_and_u8_x4.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_not_u16_x2.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_xor_u16_x2.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_or_u16_x2.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_bitwise_and_u16_x2.*}}
; SMXX-NOT: {{.*(LD\.E\.U(8|16)|PRMT|SHF|I2I|IMAD\.SHL).*}}
; SMXX: {{.*LOP3.*}}

*/
