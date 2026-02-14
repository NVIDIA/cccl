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
#include <cuda/std/limits>

#if _CCCL_HAS_LONG_DOUBLE()
constexpr bool long_double_is_double =
  cuda::std::numeric_limits<double>::min_exponent == cuda::std::numeric_limits<long double>::min_exponent
  && cuda::std::numeric_limits<double>::max_exponent == cuda::std::numeric_limits<long double>::max_exponent
  && cuda::std::numeric_limits<double>::digits == cuda::std::numeric_limits<long double>::digits
  && cuda::std::numeric_limits<double>::has_denorm == cuda::std::numeric_limits<long double>::has_denorm;
#endif // _CCCL_HAS_LONG_DOUBLE()

using fp_conv_rank_order = cuda::std::__fp_conv_rank_order;

template <class Lhs, class Rhs>
constexpr auto fp_conv_rank_order_v = cuda::std::__fp_conv_rank_order_v<Lhs, Rhs>;

static_assert(fp_conv_rank_order_v<float, float> == fp_conv_rank_order::__equal);
static_assert(fp_conv_rank_order_v<float, double> == fp_conv_rank_order::__less);
#if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<float, long double> == fp_conv_rank_order::__less);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<float, __half> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<float, __nv_bfloat16> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<float, __nv_fp8_e4m3> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<float, __nv_fp8_e5m2> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<float, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<float, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<float, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<float, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP4_E2M1()

static_assert(fp_conv_rank_order_v<double, float> == fp_conv_rank_order::__greater);
static_assert(fp_conv_rank_order_v<double, double> == fp_conv_rank_order::__equal);
#if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<double, long double> == fp_conv_rank_order::__less);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<double, __half> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<double, __nv_bfloat16> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<double, __nv_fp8_e4m3> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<double, __nv_fp8_e5m2> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<double, __nv_fp8_e8m0> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<double, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<double, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<double, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#endif // _CCCL_HAS_NVFP4_E2M1()

#if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<long double, float> == fp_conv_rank_order::__greater);
static_assert(fp_conv_rank_order_v<long double, double> == fp_conv_rank_order::__greater);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<long double, long double> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<long double, __half> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<long double, __nv_bfloat16> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<long double, __nv_fp8_e4m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<long double, __nv_fp8_e5m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<long double, __nv_fp8_e8m0> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<long double, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<long double, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<long double, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__half, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__half, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__half, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__half, __half> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__half, __nv_bfloat16> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__half, __nv_fp8_e4m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__half, __nv_fp8_e5m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__half, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__half, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__half, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__half, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_bfloat16, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __half> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_bfloat16> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp8_e4m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp8_e5m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_bfloat16, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __half> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_bfloat16> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp8_e4m3> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp8_e5m2> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp6_e2m3> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp8_e4m3, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __half> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_bfloat16> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp8_e4m3> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp8_e5m2> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp6_e2m3> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp6_e3m2> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp8_e5m2, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, float> == fp_conv_rank_order::__unordered);
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __half> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_bfloat16> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp8_e4m3> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp8_e5m2> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp8_e8m0> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp6_e2m3> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp6_e3m2> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp8_e8m0, __nv_fp4_e2m1> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __half> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_bfloat16> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp8_e4m3> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp8_e5m2> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp6_e2m3> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp6_e3m2> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp6_e2m3, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __half> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_bfloat16> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp8_e4m3> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp8_e5m2> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp6_e2m3> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp6_e3m2> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp6_e3m2, __nv_fp4_e2m1> == fp_conv_rank_order::__greater);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, float> == fp_conv_rank_order::__less);
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, double> == fp_conv_rank_order::__less);
#  if _CCCL_HAS_LONG_DOUBLE()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, long double> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_LONG_DOUBLE()
#  if _CCCL_HAS_NVFP16()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __half> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_bfloat16> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVBF16()
#  if _CCCL_HAS_NVFP8_E4M3()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp8_e4m3> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp8_e5m2> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp8_e8m0> == fp_conv_rank_order::__unordered);
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp6_e2m3> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp6_e3m2> == fp_conv_rank_order::__less);
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
static_assert(fp_conv_rank_order_v<__nv_fp4_e2m1, __nv_fp4_e2m1> == fp_conv_rank_order::__equal);
#  endif // _CCCL_HAS_NVFP4_E2M1()
#endif // _CCCL_HAS_NVFP4_E2M1()

enum class NotFloatingPoint
{
};

static_assert(fp_conv_rank_order_v<NotFloatingPoint, float> == fp_conv_rank_order::__invalid);
static_assert(fp_conv_rank_order_v<float, NotFloatingPoint> == fp_conv_rank_order::__invalid);
static_assert(fp_conv_rank_order_v<NotFloatingPoint, NotFloatingPoint> == fp_conv_rank_order::__invalid);

int main(int, char**)
{
  return 0;
}
