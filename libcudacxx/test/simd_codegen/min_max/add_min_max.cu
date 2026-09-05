//===----------------------------------------------------------------------===//
//
// Part of libcu++, the CUDA C++ Standard Library,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/simd> // IWYU pragma: keep
#include <cuda/std/cstdint>

namespace simd = cuda::std::simd;

using Vec_s32_x1 = simd::basic_vec<cuda::std::int32_t, simd::fixed_size<1>>;
using Vec_u32_x1 = simd::basic_vec<cuda::std::uint32_t, simd::fixed_size<1>>;
using Vec_s16_x2 = simd::basic_vec<cuda::std::int16_t, simd::fixed_size<2>>;
using Vec_u16_x2 = simd::basic_vec<cuda::std::uint16_t, simd::fixed_size<2>>;

__device__ Vec_s32_x1 test_add_max_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::add_max(a, b, c);
}

__device__ Vec_s32_x1 test_add_min_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::add_min(a, b, c);
}

__device__ Vec_u32_x1 test_add_max_u32(Vec_u32_x1 a, Vec_u32_x1 b, Vec_u32_x1 c)
{
  return cuda::simd::add_max(a, b, c);
}

__device__ Vec_u32_x1 test_add_min_u32(Vec_u32_x1 a, Vec_u32_x1 b, Vec_u32_x1 c)
{
  return cuda::simd::add_min(a, b, c);
}

__device__ Vec_s32_x1 test_add_max_relu_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::max_relu(a + b, c);
}

__device__ Vec_s32_x1 test_add_min_relu_s32(Vec_s32_x1 a, Vec_s32_x1 b, Vec_s32_x1 c)
{
  return cuda::simd::min_relu(a + b, c);
}

__device__ Vec_s16_x2 test_add_max_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::add_max(a, b, c);
}

__device__ Vec_s16_x2 test_add_min_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::add_min(a, b, c);
}

__device__ Vec_u16_x2 test_add_max_u16_x2(Vec_u16_x2 a, Vec_u16_x2 b, Vec_u16_x2 c)
{
  return cuda::simd::add_max(a, b, c);
}

__device__ Vec_u16_x2 test_add_min_u16_x2(Vec_u16_x2 a, Vec_u16_x2 b, Vec_u16_x2 c)
{
  return cuda::simd::add_min(a, b, c);
}

__device__ Vec_s16_x2 test_add_max_relu_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::max_relu(a + b, c);
}

__device__ Vec_s16_x2 test_add_min_relu_s16_x2(Vec_s16_x2 a, Vec_s16_x2 b, Vec_s16_x2 c)
{
  return cuda::simd::min_relu(a + b, c);
}

/*

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_relu_s16_x2.*}}
; SM90: {{.*VIADDMNMX\.S16x2\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX\.S16x2\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX\.S16x2\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_relu_s16_x2.*}}
; SM90: {{.*VIADDMNMX\.S16x2\.RELU.*!PT.*}}
; SM100: {{.*VIADDMNMX\.S16x2\.RELU.*!PT.*}}
; SM103: {{.*VIADDMNMX\.S16x2\.RELU.*!PT.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.S16x2\.RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_u16_x2.*}}
; SM90: {{.*VIADDMNMX\.U16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX\.U16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX\.U16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.U16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.U16x2.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_u16_x2.*}}
; SM90: {{.*VIADDMNMX\.U16x2.*!PT.*}}
; SM100: {{.*VIADDMNMX\.U16x2.*!PT.*}}
; SM103: {{.*VIADDMNMX\.U16x2.*!PT.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.U16x2.*!PT.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.U16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_s16_x2.*}}
; SM90: {{.*VIADDMNMX\.S16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX\.S16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX\.S16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.S16x2.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.S16x2.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_s16_x2.*}}
; SM90: {{.*VIADDMNMX\.S16x2.*!PT.*}}
; SM100: {{.*VIADDMNMX\.S16x2.*!PT.*}}
; SM103: {{.*VIADDMNMX\.S16x2.*!PT.*}}
; SM107: {{.*VIADD\.16x2.*}}
; SM107: {{.*VIMNMX\.S16x2.*!PT.*}}
; SM120: {{.*VIADD\.16x2.*}}
; SM120: {{.*VIMNMX\.S16x2.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_relu_s32.*}}
; SM90: {{.*VIADDMNMX\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.S32\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.S32\.RELU.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_relu_s32.*}}
; SM90: {{.*VIADDMNMX\.RELU.*!PT.*}}
; SM100: {{.*VIADDMNMX\.RELU.*!PT.*}}
; SM103: {{.*VIADDMNMX\.RELU.*!PT.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.S32\.RELU.*!PT.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.S32\.RELU.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_u32.*}}
; SM90: {{.*VIADDMNMX\.U32.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX\.U32.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX\.U32.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.U32.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.U32.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_u32.*}}
; SM90: {{.*VIADDMNMX\.U32.*!PT.*}}
; SM100: {{.*VIADDMNMX\.U32.*!PT.*}}
; SM103: {{.*VIADDMNMX\.U32.*!PT.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.U32.*!PT.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.U32.*!PT.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_min_s32.*}}
; SM90: {{.*VIADDMNMX.*,[[:space:]]PT[[:space:]]*;.*}}
; SM100: {{.*VIADDMNMX.*,[[:space:]]PT[[:space:]]*;.*}}
; SM103: {{.*VIADDMNMX.*,[[:space:]]PT[[:space:]]*;.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.S32.*,[[:space:]]PT[[:space:]]*;.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.S32.*,[[:space:]]PT[[:space:]]*;.*}}

; SMXX-LABEL: {{[[:space:]]*}}Function : {{.*test_add_max_s32.*}}
; SM90: {{.*VIADDMNMX.*!PT.*}}
; SM100: {{.*VIADDMNMX.*!PT.*}}
; SM103: {{.*VIADDMNMX.*!PT.*}}
; SM107: {{.*IADD.*}}
; SM107: {{.*VIMNMX\.S32.*!PT.*}}
; SM120: {{.*IADD.*}}
; SM120: {{.*VIMNMX\.S32.*!PT.*}}

*/
