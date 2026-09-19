// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/complex>

// Give the inspected parameter a stack location that survives optimization, so the
// debugger can read it in this frame. Without this the parameter stays in a
// caller-clobbered register and reads as unavailable at -O3.
#define KEEP_FOR_DEBUGGER(values) asm volatile("" : : "g"(&(values)) : "memory")

using complex_alias = cuda::std::complex<float>;

[[gnu::noinline]] void inspect_std_default(const cuda::std::complex<float>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_std_double(const cuda::std::complex<double>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_std_half(const cuda::std::complex<__half>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_std_bfloat16(const cuda::std::complex<__nv_bfloat16>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_cuda_float(const cuda::complex<float>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_alias(const complex_alias& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::complex<float>, 2>& values)
{
  KEEP_FOR_DEBUGGER(values);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::complex<double>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::complex<double>& value)
{
  KEEP_FOR_DEBUGGER(value);
}

int main()
{
  const cuda::std::complex<float> std_default{};
  const cuda::std::complex<double> std_double{1.5, -2.25};
  const cuda::std::complex<__half> std_half{::__float2half(1.5f), ::__float2half(-2.25f)};
  const cuda::std::complex<__nv_bfloat16> std_bfloat16{::__float2bfloat16(1.5f), ::__float2bfloat16(-2.25f)};
  const cuda::complex<float> cuda_float{0.5f, 42.5f};
  const complex_alias alias{-3.5f, 0.25f};
  const cuda::std::array<cuda::std::complex<float>, 2> nested = {{{13.5f, -5.5f}, {0.25f, 88.5f}}};
  cuda::std::complex<double> updated{6.5, -91.5};

  inspect_std_default(std_default);
  inspect_std_double(std_double);
  inspect_std_half(std_half);
  inspect_std_bfloat16(std_bfloat16);
  inspect_cuda_float(cuda_float);
  inspect_alias(alias);
  inspect_nested(nested);
  inspect_before_update(updated);
  updated = {3.25, 85.5};
  inspect_after_update(updated);
}
