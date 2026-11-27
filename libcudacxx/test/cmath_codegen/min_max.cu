//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The arguments are loaded from global memory so that the checks catch min and max selecting the address of an
// argument and loading the result from it, instead of selecting the loaded value.

#include <cuda/cmath>
#include <cuda/std/cstdint>

__global__ void test_min_s32(cuda::std::int32_t* out, const cuda::std::int32_t* a, const cuda::std::int32_t* b)
{
  *out = cuda::min(*a, *b);
}

__global__ void test_max_s32(cuda::std::int32_t* out, const cuda::std::int32_t* a, const cuda::std::int32_t* b)
{
  *out = cuda::max(*a, *b);
}

__global__ void test_min_f32(float* out, const float* a, const float* b)
{
  *out = cuda::min(*a, *b);
}

__global__ void test_max_f32(float* out, const float* a, const float* b)
{
  *out = cuda::max(*a, *b);
}

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>

__global__ void test_min_f16(__half* out, const __half* a, const __half* b)
{
  *out = cuda::min(*a, *b);
}

__global__ void test_max_f16(__half* out, const __half* a, const __half* b)
{
  *out = cuda::max(*a, *b);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>

__global__ void test_min_bf16(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b)
{
  *out = cuda::min(*a, *b);
}

__global__ void test_max_bf16(__nv_bfloat16* out, const __nv_bfloat16* a, const __nv_bfloat16* b)
{
  *out = cuda::max(*a, *b);
}
#endif // _CCCL_HAS_NVBF16()

/*

; SMXX-LABEL: .visible .entry {{.*}}test_min_s32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}min.s32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_max_s32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}max.s32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_min_f32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}min.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_max_f32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}max.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_min_f16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|cvt\.}}
; SM80-PLUS: {{.*}}min.f16 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}min.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_max_f16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|cvt\.}}
; SM80-PLUS: {{.*}}max.f16 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}max.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_min_bf16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|mov\.b32}}
; SM80-PLUS: {{.*}}min.bf16 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}min.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_max_bf16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|mov\.b32}}
; SM80-PLUS: {{.*}}max.bf16 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}max.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|ld\.global}}
; SMXX: ret;

*/
