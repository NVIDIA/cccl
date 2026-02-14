//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// <cuda/std/numbers>

#include <cuda/std/cassert>
#include <cuda/std/numbers>

#include <test_macros.h>

TEST_NV_DIAG_SUPPRESS(1046) // Suppress "floating-point value cannot be represented exactly"

__host__ __device__ constexpr bool test()
{
  // default constants
  assert(cuda::std::numbers::e == 2.718281828459045);
  assert(cuda::std::numbers::log2e == 1.4426950408889634);
  assert(cuda::std::numbers::log10e == 0.4342944819032518);
  assert(cuda::std::numbers::pi == 3.141592653589793);
  assert(cuda::std::numbers::inv_pi == 0.3183098861837907);
  assert(cuda::std::numbers::inv_sqrtpi == 0.5641895835477563);
  assert(cuda::std::numbers::ln2 == 0.6931471805599453);
  assert(cuda::std::numbers::ln10 == 2.302585092994046);
  assert(cuda::std::numbers::sqrt2 == 1.4142135623730951);
  assert(cuda::std::numbers::sqrt3 == 1.7320508075688772);
  assert(cuda::std::numbers::inv_sqrt3 == 0.5773502691896257);
  assert(cuda::std::numbers::egamma == 0.5772156649015329);
  assert(cuda::std::numbers::phi == 1.618033988749895);

  // float constants
  assert(cuda::std::numbers::e_v<float> == 2.7182817f);
  assert(cuda::std::numbers::log2e_v<float> == 1.442695f);
  assert(cuda::std::numbers::log10e_v<float> == 0.4342945f);
  assert(cuda::std::numbers::pi_v<float> == 3.1415927f);
  assert(cuda::std::numbers::inv_pi_v<float> == 0.31830987f);
  assert(cuda::std::numbers::inv_sqrtpi_v<float> == 0.5641896f);
  assert(cuda::std::numbers::ln2_v<float> == 0.6931472f);
  assert(cuda::std::numbers::ln10_v<float> == 2.3025851f);
  assert(cuda::std::numbers::sqrt2_v<float> == 1.4142135f);
  assert(cuda::std::numbers::sqrt3_v<float> == 1.7320508f);
  assert(cuda::std::numbers::inv_sqrt3_v<float> == 0.57735026f);
  assert(cuda::std::numbers::egamma_v<float> == 0.5772157f);
  assert(cuda::std::numbers::phi_v<float> == 1.618034f);

  // double constants
  assert(cuda::std::numbers::e_v<double> == 2.718281828459045);
  assert(cuda::std::numbers::log2e_v<double> == 1.4426950408889634);
  assert(cuda::std::numbers::log10e_v<double> == 0.4342944819032518);
  assert(cuda::std::numbers::pi_v<double> == 3.141592653589793);
  assert(cuda::std::numbers::inv_pi_v<double> == 0.3183098861837907);
  assert(cuda::std::numbers::inv_sqrtpi_v<double> == 0.5641895835477563);
  assert(cuda::std::numbers::ln2_v<double> == 0.6931471805599453);
  assert(cuda::std::numbers::ln10_v<double> == 2.302585092994046);
  assert(cuda::std::numbers::sqrt2_v<double> == 1.4142135623730951);
  assert(cuda::std::numbers::sqrt3_v<double> == 1.7320508075688772);
  assert(cuda::std::numbers::inv_sqrt3_v<double> == 0.5773502691896257);
  assert(cuda::std::numbers::egamma_v<double> == 0.5772156649015329);
  assert(cuda::std::numbers::phi_v<double> == 1.618033988749895);

  // fixme: this supposes that long double format is fp80_x64, should be generic
#if _CCCL_HAS_LONG_DOUBLE()
  // long double constants
  assert(cuda::std::numbers::e_v<long double> == 2.7182818284590452354l);
  assert(cuda::std::numbers::log2e_v<long double> == 1.4426950408889634074l);
  assert(cuda::std::numbers::log10e_v<long double> == 0.43429448190325182765l);
  assert(cuda::std::numbers::pi_v<long double> == 3.1415926535897932385l);
  assert(cuda::std::numbers::inv_pi_v<long double> == 0.31830988618379067154l);
  assert(cuda::std::numbers::inv_sqrtpi_v<long double> == 0.5641895835477562869l);
  assert(cuda::std::numbers::ln2_v<long double> == 0.69314718055994530943l);
  assert(cuda::std::numbers::ln10_v<long double> == 2.302585092994045684l);
  assert(cuda::std::numbers::sqrt2_v<long double> == 1.4142135623730950488l);
  assert(cuda::std::numbers::sqrt3_v<long double> == 1.7320508075688772936l);
  assert(cuda::std::numbers::inv_sqrt3_v<long double> == 0.5773502691896257645l);
  assert(cuda::std::numbers::egamma_v<long double> == 0.5772156649015328606l);
  assert(cuda::std::numbers::phi_v<long double> == 1.6180339887498948482l);
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_FLOAT128()
  assert(cuda::std::numbers::e_v<__float128> == 2.7182818284590452353602874713526623q);
  assert(cuda::std::numbers::log2e_v<__float128> == 1.442695040888963407359924681001892q);
  assert(cuda::std::numbers::log10e_v<__float128> == 0.4342944819032518276511289189166051q);
  assert(cuda::std::numbers::pi_v<__float128> == 3.1415926535897932384626433832795028q);
  assert(cuda::std::numbers::inv_pi_v<__float128> == 0.31830988618379067153776752674502874q);
  assert(cuda::std::numbers::inv_sqrtpi_v<__float128> == 0.5641895835477562869480794515607726q);
  assert(cuda::std::numbers::ln2_v<__float128> == 0.6931471805599453094172321214581766q);
  assert(cuda::std::numbers::ln10_v<__float128> == 2.302585092994045684017991454684364q);
  assert(cuda::std::numbers::sqrt2_v<__float128> == 1.414213562373095048801688724209698q);
  assert(cuda::std::numbers::sqrt3_v<__float128> == 1.7320508075688772935274463415058723q);
  assert(cuda::std::numbers::inv_sqrt3_v<__float128> == 0.5773502691896257645091487805019574q);
  assert(cuda::std::numbers::egamma_v<__float128> == 0.5772156649015328606065120900824025q);
  assert(cuda::std::numbers::phi_v<__float128> == 1.6180339887498948482045868343656382q);
#endif // _CCCL_HAS_FLOAT128()

  return true;
}

// Extended floating point types are not comparable in constexpr context
__host__ __device__ void test_ext_fp()
{
#if !TEST_COMPILER(MSVC)
  // MSVC errors here because of "error: A __device__ variable template cannot have a const qualified type on Windows"
#  if _LIBCUDACXX_HAS_NVFP16()
  // __half constants
  assert(cuda::std::numbers::e_v<__half> == __half{2.7182817f});
  assert(cuda::std::numbers::log2e_v<__half> == __half{1.442695f});
  assert(cuda::std::numbers::log10e_v<__half> == __half{0.4342945f});
  assert(cuda::std::numbers::pi_v<__half> == __half{3.1415927f});
  assert(cuda::std::numbers::inv_pi_v<__half> == __half{0.31830987f});
  assert(cuda::std::numbers::inv_sqrtpi_v<__half> == __half{0.5641896f});
  assert(cuda::std::numbers::ln2_v<__half> == __half{0.6931472f});
  assert(cuda::std::numbers::ln10_v<__half> == __half{2.3025851f});
  assert(cuda::std::numbers::sqrt2_v<__half> == __half{1.4142135f});
  assert(cuda::std::numbers::sqrt3_v<__half> == __half{1.7320508f});
  assert(cuda::std::numbers::inv_sqrt3_v<__half> == __half{0.57735026f});
  assert(cuda::std::numbers::egamma_v<__half> == __half{0.5772157f});
  assert(cuda::std::numbers::phi_v<__half> == __half{1.618034f});
#  endif // _LIBCUDACXX_HAS_NVFP16()

#  if _LIBCUDACXX_HAS_NVBF16()
  assert(cuda::std::numbers::e_v<__nv_bfloat16> == __nv_bfloat16{2.7182817f});
  assert(cuda::std::numbers::log2e_v<__nv_bfloat16> == __nv_bfloat16{1.442695f});
  assert(cuda::std::numbers::log10e_v<__nv_bfloat16> == __nv_bfloat16{0.4342945f});
  assert(cuda::std::numbers::pi_v<__nv_bfloat16> == __nv_bfloat16{3.1415927f});
  assert(cuda::std::numbers::inv_pi_v<__nv_bfloat16> == __nv_bfloat16{0.31830987f});
  assert(cuda::std::numbers::inv_sqrtpi_v<__nv_bfloat16> == __nv_bfloat16{0.5641896f});
  assert(cuda::std::numbers::ln2_v<__nv_bfloat16> == __nv_bfloat16{0.6931472f});
  assert(cuda::std::numbers::ln10_v<__nv_bfloat16> == __nv_bfloat16{2.3025851f});
  assert(cuda::std::numbers::sqrt2_v<__nv_bfloat16> == __nv_bfloat16{1.4142135f});
  assert(cuda::std::numbers::sqrt3_v<__nv_bfloat16> == __nv_bfloat16{1.7320508f});
  assert(cuda::std::numbers::inv_sqrt3_v<__nv_bfloat16> == __nv_bfloat16{0.57735026f});
  assert(cuda::std::numbers::egamma_v<__nv_bfloat16> == __nv_bfloat16{0.5772157f});
  assert(cuda::std::numbers::phi_v<__nv_bfloat16> == __nv_bfloat16{1.618034f});
#  endif // _LIBCUDACXX_HAS_NVBF16()
#endif // !TEST_COMPILER(MSVC)
}

__global__ void test_kernel()
{
  test();
  test_ext_fp();
}

int main(int, char**)
{
  test();
  test_ext_fp();
  static_assert(test());
  return 0;
}
