//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>

#include "test_macros.h"

// binary16

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__binary16>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__binary16> == 5);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__binary16> == 15);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__binary16> == -14);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__binary16> == 15);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__binary16> == 10);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__binary16>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__binary16> == 11);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__binary16>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__binary16>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__binary16>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__binary16>);

// binary32

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__binary32>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__binary32> == 8);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__binary32> == 127);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__binary32> == -126);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__binary32> == 127);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__binary32> == 23);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__binary32>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__binary32> == 24);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__binary32>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__binary32>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__binary32>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__binary32>);

// binary64

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__binary64>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__binary64> == 11);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__binary64> == 1023);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__binary64> == -1022);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__binary64> == 1023);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__binary64> == 52);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__binary64>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__binary64> == 53);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__binary64>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__binary64>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__binary64>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__binary64>);

// binary128

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__binary128>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__binary128> == 15);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__binary128> == 16383);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__binary128> == -16382);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__binary128> == 16383);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__binary128> == 112);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__binary128>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__binary128> == 113);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__binary128>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__binary128>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__binary128>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__binary128>);

// bfloat16

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__bfloat16>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__bfloat16> == 8);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__bfloat16> == 127);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__bfloat16> == -126);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__bfloat16> == 127);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__bfloat16> == 7);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__bfloat16>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__bfloat16> == 8);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__bfloat16>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__bfloat16>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__bfloat16>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__bfloat16>);

// fp80_x86

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp80_x86>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp80_x86> == 15);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp80_x86> == 16383);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp80_x86> == -16382);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp80_x86> == 16383);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp80_x86> == 64);
static_assert(!cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp80_x86>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp80_x86> == 64);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp80_x86>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp80_x86>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp80_x86>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp80_x86>);

// fp8_nv_e4m3

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp8_nv_e4m3>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp8_nv_e4m3> == 4);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp8_nv_e4m3> == 7);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp8_nv_e4m3> == -6);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp8_nv_e4m3> == 8);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp8_nv_e4m3> == 3);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp8_nv_e4m3>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp8_nv_e4m3> == 4);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp8_nv_e4m3>);
static_assert(!cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp8_nv_e4m3>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp8_nv_e4m3>);
static_assert(!cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp8_nv_e4m3>);

// fp8_nv_e5m2

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp8_nv_e5m2>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp8_nv_e5m2> == 5);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp8_nv_e5m2> == 15);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp8_nv_e5m2> == -14);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp8_nv_e5m2> == 15);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp8_nv_e5m2> == 2);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp8_nv_e5m2>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp8_nv_e5m2> == 3);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp8_nv_e5m2>);
static_assert(cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp8_nv_e5m2>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp8_nv_e5m2>);
static_assert(cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp8_nv_e5m2>);

// fp8_nv_e8m0

static_assert(!cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp8_nv_e8m0>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp8_nv_e8m0> == 8);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp8_nv_e8m0> == 127);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp8_nv_e8m0> == -127);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp8_nv_e8m0> == 127);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp8_nv_e8m0> == 0);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp8_nv_e8m0>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp8_nv_e8m0> == 1);
static_assert(!cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp8_nv_e8m0>);
static_assert(!cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp8_nv_e8m0>);
static_assert(cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp8_nv_e8m0>);
static_assert(!cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp8_nv_e8m0>);

// fp6_nv_e2m3

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp6_nv_e2m3>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 2);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 1);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 0);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 2);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 3);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp6_nv_e2m3>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp6_nv_e2m3> == 4);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp6_nv_e2m3>);
static_assert(!cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp6_nv_e2m3>);
static_assert(!cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp6_nv_e2m3>);
static_assert(!cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp6_nv_e2m3>);

// fp6_nv_e3m2

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp6_nv_e3m2>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp6_nv_e3m2> == 3);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp6_nv_e3m2> == 3);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp6_nv_e3m2> == -2);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp6_nv_e3m2> == 4);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp6_nv_e3m2> == 2);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp6_nv_e3m2>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp6_nv_e3m2> == 3);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp6_nv_e3m2>);
static_assert(!cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp6_nv_e3m2>);
static_assert(!cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp6_nv_e3m2>);
static_assert(!cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp6_nv_e3m2>);

// fp4_nv_e2m1

static_assert(cuda::std::__fp_is_signed_v<cuda::std::__fp_format::__fp4_nv_e2m1>);
static_assert(cuda::std::__fp_exp_nbits_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 2);
static_assert(cuda::std::__fp_exp_bias_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 1);
static_assert(cuda::std::__fp_exp_min_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 0);
static_assert(cuda::std::__fp_exp_max_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 2);
static_assert(cuda::std::__fp_mant_nbits_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 1);
static_assert(cuda::std::__fp_has_implicit_bit_v<cuda::std::__fp_format::__fp4_nv_e2m1>);
static_assert(cuda::std::__fp_digits_v<cuda::std::__fp_format::__fp4_nv_e2m1> == 2);
static_assert(cuda::std::__fp_has_denorm_v<cuda::std::__fp_format::__fp4_nv_e2m1>);
static_assert(!cuda::std::__fp_has_inf_v<cuda::std::__fp_format::__fp4_nv_e2m1>);
static_assert(!cuda::std::__fp_has_nan_v<cuda::std::__fp_format::__fp4_nv_e2m1>);
static_assert(!cuda::std::__fp_has_nans_v<cuda::std::__fp_format::__fp4_nv_e2m1>);

int main(int, char**)
{
  return 0;
}
