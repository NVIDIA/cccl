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

using cuda::std::__fp_format;
using cuda::std::__fp_is_subset_v;

static_assert(__fp_is_subset_v<__fp_format::__binary16, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__binary16, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__binary16, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__binary16, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary16, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__binary32, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__binary32, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__binary32, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary32, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__binary16>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__binary64, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__binary64, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary64, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__binary16>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__binary32>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__binary128, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__bfloat16>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__binary128, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__bfloat16, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__binary16>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__binary32>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp80_x86, __fp_format::__fp4_nv_e2m1>);

static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp80_x86>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e4m3, __fp_format::__fp4_nv_e2m1>);

static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp8_nv_e4m3>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e5m2, __fp_format::__fp4_nv_e2m1>);

static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__binary16>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__binary128>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp80_x86>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp8_nv_e5m2>);
static_assert(__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp8_nv_e8m0, __fp_format::__fp4_nv_e2m1>);

static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp80_x86>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp8_nv_e4m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp8_nv_e8m0>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp6_nv_e2m3>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e2m3, __fp_format::__fp4_nv_e2m1>);

static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp80_x86>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp8_nv_e4m3>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp8_nv_e8m0>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp6_nv_e2m3>);
static_assert(__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp6_nv_e3m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp6_nv_e3m2, __fp_format::__fp4_nv_e2m1>);

static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__binary16>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__binary32>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__binary64>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__binary128>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__bfloat16>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp80_x86>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp8_nv_e4m3>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp8_nv_e5m2>);
static_assert(!__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp8_nv_e8m0>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp6_nv_e2m3>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp6_nv_e3m2>);
static_assert(__fp_is_subset_v<__fp_format::__fp4_nv_e2m1, __fp_format::__fp4_nv_e2m1>);

int main(int, char**)
{
  return 0;
}
