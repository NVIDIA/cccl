//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/cfloat>

#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  // FLT_RADIX
  static_assert(FLT_RADIX == 2);

  // DECIMAL_DIG
  [[maybe_unused]] constexpr auto decimal_dig = DECIMAL_DIG;

  // FLT_ROUNDS
  [[maybe_unused]] constexpr auto flt_rounds = FLT_ROUNDS;

  // FLT_EVAL_METHOD
  [[maybe_unused]] constexpr auto flt_eval_method = FLT_EVAL_METHOD;

  // FLT_DECIMAL_DIG
  static_assert(FLT_DECIMAL_DIG == 9);

  // FLT_MIN
  static_assert(FLT_MIN == 1.17549435082228750796873653722224568e-38f);
  static_assert(cuda::std::is_same_v<decltype(FLT_MIN), float>);

  // FLT_TRUE_MIN
  static_assert(FLT_TRUE_MIN == 1.40129846432481707092372958328991613e-45f);
  static_assert(cuda::std::is_same_v<decltype(FLT_TRUE_MIN), float>);

  // FLT_MAX
  static_assert(FLT_MAX == 3.40282346638528859811704183484516925e+38f);
  static_assert(cuda::std::is_same_v<decltype(FLT_MAX), float>);

  // FLT_EPSILON
  static_assert(FLT_EPSILON == 1.19209289550781250000000000000000000e-7f);
  static_assert(cuda::std::is_same_v<decltype(FLT_EPSILON), float>);

  // FLT_DIG
  static_assert(FLT_DIG == 6);

  // FLT_MANT_DIG
  static_assert(FLT_MANT_DIG == 24);

  // FLT_MIN_EXP
  static_assert(FLT_MIN_EXP == -125);

  // FLT_MIN_10_EXP
  static_assert(FLT_MIN_10_EXP == -37);

  // FLT_MAX_EXP
  static_assert(FLT_MAX_EXP == 128);

  // FLT_MAX_10_EXP
  static_assert(FLT_MAX_10_EXP == 38);

  // FLT_HAS_SUBNORM
  static_assert(FLT_HAS_SUBNORM == 1);

  // DBL_DECIMAL_DIG
  static_assert(DBL_DECIMAL_DIG == 17);

  // DBL_MIN
  static_assert(DBL_MIN == 2.22507385850720138309023271733240406e-308);
  static_assert(cuda::std::is_same_v<decltype(DBL_MIN), double>);

  // DBL_TRUE_MIN
  static_assert(DBL_TRUE_MIN == 4.94065645841246544176568792868221372e-324);
  static_assert(cuda::std::is_same_v<decltype(DBL_TRUE_MIN), double>);

  // DBL_MAX
  static_assert(DBL_MAX == 1.79769313486231570814527423731704357e+308);
  static_assert(cuda::std::is_same_v<decltype(DBL_MAX), double>);

  // DBL_EPSILON
  static_assert(DBL_EPSILON == 2.22044604925031308084726333618164062e-16);
  static_assert(cuda::std::is_same_v<decltype(DBL_EPSILON), double>);

  // DBL_DIG
  static_assert(DBL_DIG == 15);

  // DBL_MANT_DIG
  static_assert(DBL_MANT_DIG == 53);

  // DBL_MIN_EXP
  static_assert(DBL_MIN_EXP == -1021);

  // DBL_MIN_10_EXP
  static_assert(DBL_MIN_10_EXP == -307);

  // DBL_MAX_EXP
  static_assert(DBL_MAX_EXP == 1024);

  // DBL_MAX_10_EXP
  static_assert(DBL_MAX_10_EXP == 308);

  // DBL_HAS_SUBNORM
  static_assert(DBL_HAS_SUBNORM == 1);

  // LDBL_*
#if _CCCL_HAS_LONG_DOUBLE()
  [[maybe_unused]] constexpr auto ldbl_decimal_dig = LDBL_DECIMAL_DIG;
  [[maybe_unused]] constexpr auto ldbl_min         = LDBL_MIN;
  [[maybe_unused]] constexpr auto ldbl_true_min    = LDBL_TRUE_MIN;
  [[maybe_unused]] constexpr auto ldbl_max         = LDBL_MAX;
  [[maybe_unused]] constexpr auto ldbl_epsilon     = LDBL_EPSILON;
  [[maybe_unused]] constexpr auto ldbl_dig         = LDBL_DIG;
  [[maybe_unused]] constexpr auto ldbl_mant_dig    = LDBL_MANT_DIG;
  [[maybe_unused]] constexpr auto ldbl_min_exp     = LDBL_MIN_EXP;
  [[maybe_unused]] constexpr auto ldbl_min_10_exp  = LDBL_MIN_10_EXP;
  [[maybe_unused]] constexpr auto ldbl_max_exp     = LDBL_MAX_EXP;
  [[maybe_unused]] constexpr auto ldbl_max_10_exp  = LDBL_MAX_10_EXP;
  [[maybe_unused]] constexpr auto ldbl_has_subnorm = LDBL_HAS_SUBNORM;
  [[maybe_unused]] constexpr auto ldbl_min_10_exp  = LDBL_MIN_10_EXP;
  [[maybe_unused]] constexpr auto ldbl_max_10_exp  = LDBL_MAX_10_EXP;
  [[maybe_unused]] constexpr auto ldbl_has_subnorm = LDBL_HAS_SUBNORM;
#endif // _CCCL_HAS_LONG_DOUBLE()

  return 0;
}
