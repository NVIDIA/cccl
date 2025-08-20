//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/cuda_fp_types.h>
#include <cuda/std/cassert>
#include <cuda/std/limits>

__device__ void test()
{
#if _CCCL_HAS_FLOAT128() && _CCCL_DEVICE_COMPILATION() && _CCCL_CTK_AT_LEAST(12, 8)
  __float128 dummy_f128{};
  int dummy_int{};

  assert(__nv_fp128_sqrt(1.q) == 1.q);
  assert(__nv_fp128_sin(0.q) == 0.q);
  assert(__nv_fp128_cos(0.q) == 1.q);
  assert(__nv_fp128_tan(0.q) == 0.q);
  assert(__nv_fp128_asin(0.q) == 0.q);
  assert(__nv_fp128_acos(1.q) == 0.q);
  assert(__nv_fp128_atan(0.q) == 0.q);
  assert(__nv_fp128_exp(0.q) == 1.q);
  assert(__nv_fp128_exp2(0.q) == 1.q);
  assert(__nv_fp128_exp10(0.q) == 1.q);
  assert(__nv_fp128_expm1(0.q) == 0.q);
  assert(__nv_fp128_log(1.q) == 0.q);
  assert(__nv_fp128_log2(1.q) == 0.q);
  assert(__nv_fp128_log10(1.q) == 0.q);
  assert(__nv_fp128_log1p(0.q) == 0.q);
  assert(__nv_fp128_pow(1.q, 1.q) == 1.q);
  assert(__nv_fp128_sinh(0.q) == 0.q);
  assert(__nv_fp128_cosh(0.q) == 1.q);
  assert(__nv_fp128_tanh(0.q) == 0.q);
  assert(__nv_fp128_asinh(0.q) == 0.q);
  assert(__nv_fp128_acosh(1.q) == 0.q);
  assert(__nv_fp128_atanh(0.q) == 0.q);
  assert(__nv_fp128_trunc(1.5q) == 1.q);
  assert(__nv_fp128_floor(1.5q) == 1.q);
  assert(__nv_fp128_ceil(1.5q) == 2.q);
  assert(__nv_fp128_round(1.5q) == 2.q);
  assert(__nv_fp128_rint(1.99q) == 2.q);
  assert(__nv_fp128_fabs(-1.q) == 1.q);
  assert(__nv_fp128_copysign(1.q, -1.q) == -1.q);
  assert(__nv_fp128_fmax(1.q, -1.q) == 1.q);
  assert(__nv_fp128_fmin(1.q, -1.q) == -1.q);
  assert(__nv_fp128_fdim(1.q, 1.q) == 0.q);
  assert(__nv_fp128_fmod(1.q, 1.q) == 0.q);
  assert(__nv_fp128_remainder(1.q, 1.q) == 0.q);
  assert(__nv_fp128_frexp(1.q, &dummy_int) == 1.q);
  assert(__nv_fp128_modf(1.q, &dummy_f128) == 1.q);
  assert(__nv_fp128_hypot(3.q, 4.q) == 5.q);
  assert(__nv_fp128_fma(1.q, 1.q, 1.q) == 2.q);
  assert(__nv_fp128_ldexp(1.q, 0) == 1.q);
  assert(__nv_fp128_ilogb(cuda::std::numeric_limits<__float128>::infinity()) == cuda::std::numeric_limits<int>::max());
  assert(__nv_fp128_mul(1.q, 1.q) == 1.q);
  assert(__nv_fp128_add(1.q, 1.q) == 2.q);
  assert(__nv_fp128_sub(1.q, 1.q) == 0.q);
  assert(__nv_fp128_div(1.q, 1.q) == 1.q);
  assert(__nv_fp128_isnan(1.q) == false);
  assert(__nv_fp128_isunordered(1.q, 1.q) == false);
#endif // _CCCL_HAS_FLOAT128() && _CCCL_DEVICE_COMPILATION() && _CCCL_CTK_AT_LEAST(12, 8)
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
