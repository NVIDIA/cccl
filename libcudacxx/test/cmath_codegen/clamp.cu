//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The arguments are loaded from global memory so that the checks catch clamp selecting the address of an argument and
// loading the result from it, instead of selecting the loaded value.

#include <cuda/cmath>
#include <cuda/std/cstdint>

__global__ void test_clamp_s32(
  cuda::std::int32_t* out, const cuda::std::int32_t* v, const cuda::std::int32_t* lo, const cuda::std::int32_t* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}

__global__ void test_clamp_u64(
  cuda::std::uint64_t* out, const cuda::std::uint64_t* v, const cuda::std::uint64_t* lo, const cuda::std::uint64_t* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}

__global__ void test_clamp_f32(float* out, const float* v, const float* lo, const float* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}

// On GPUs with slow FP64 arithmetic, cuda::std::fmin and cuda::std::fmax compare double with integer instructions
__global__ void test_clamp_f64(double* out, const double* v, const double* lo, const double* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>

__global__ void test_clamp_f16(__half* out, const __half* v, const __half* lo, const __half* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#  include <cuda_bf16.h>

__global__ void
test_clamp_bf16(__nv_bfloat16* out, const __nv_bfloat16* v, const __nv_bfloat16* lo, const __nv_bfloat16* hi)
{
  *out = cuda::clamp(*v, *lo, *hi);
}
#endif // _CCCL_HAS_NVBF16()

/*

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_s32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}min.s32 [[S32_MIN:%[a-z]+[0-9]+]], {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}max.s32 {{%[a-z]+[0-9]+}}, [[S32_MIN]], {{%[a-z]+[0-9]+}};
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_u64{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}min.u64 [[U64_MIN:%[a-z]+[0-9]+]], {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}max.u64 {{%[a-z]+[0-9]+}}, [[U64_MIN]], {{%[a-z]+[0-9]+}};
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_f32{{.*}}(
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}min.f32 [[F32_MIN:%[a-z]+[0-9]+]], {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SMXX: {{.*}}max.f32 {{%[a-z]+[0-9]+}}, [[F32_MIN]], {{%[a-z]+[0-9]+}};
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_f64{{.*}}(
; SMXX-COUNT-3: {{.*}}ld.global.{{[bfu]}}64 {{.*}}
; SM80-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM80: {{.*}}min.f64 [[F64_MIN:%[a-z]+[0-9]+]], {{.*}}
; SM80-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM80: {{.*}}max.f64 {{%[a-z]+[0-9]+}}, [[F64_MIN]], {{%[a-z]+[0-9]+}};
; SM90-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM90: {{.*}}min.f64 [[F64_MIN:%[a-z]+[0-9]+]], {{.*}}
; SM90-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM90: {{.*}}max.f64 {{%[a-z]+[0-9]+}}, [[F64_MIN]], {{%[a-z]+[0-9]+}};
; SM100-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM100: {{.*}}min.f64 [[F64_MIN:%[a-z]+[0-9]+]], {{.*}}
; SM100-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; SM100: {{.*}}max.f64 {{%[a-z]+[0-9]+}}, [[F64_MIN]], {{%[a-z]+[0-9]+}};
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_f16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|cvt\.}}
; SM80-PLUS: {{.*}}min.f16 [[F16_MIN:%[a-z]+[0-9]+]],{{.*}}
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|cvt\.}}
; SM80-PLUS: {{.*}}max.f16 {{%[a-z]+[0-9]+}},[[F16_MIN]],{{%[a-z]+[0-9]+}};
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}min.f32 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}max.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

; SMXX-LABEL: .visible .entry {{.*}}test_clamp_bf16{{.*}}(
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|mov\.b32}}
; SM80-PLUS: {{.*}}min.bf16 [[BF16_MIN:%[a-z]+[0-9]+]],{{.*}}
; SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp|mov\.b32}}
; SM80-PLUS: {{.*}}max.bf16 {{%[a-z]+[0-9]+}},[[BF16_MIN]],{{%[a-z]+[0-9]+}};
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}min.f32 {{.*}}
; NOT-SM80-PLUS-NOT: {{[[:space:]]bra[[:space:].]|setp|selp}}
; NOT-SM80-PLUS: {{.*}}max.f32 {{.*}}
; SMXX-NOT: {{[[:space:]]bra[[:space:].]|ld\.global}}
; SMXX: ret;

*/
