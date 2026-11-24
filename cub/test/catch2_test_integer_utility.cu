// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/integer_utility.cuh>

#include <cuda/std/__floating_point/arithmetic.h>
#include <cuda/std/__floating_point/constants.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/limits>

#include "cub_test_macros.h"
#include <c2h/extended_types.h>

// clang-format off
using fp_types = c2h::type_list<
  float,
  double
#if TEST_HALF_T()
  , __half
#endif // TEST_HALF_T()
#if TEST_BF_T()
  , __nv_bfloat16
#endif // TEST_BF_T()
#if _CCCL_HAS_FLOAT128()
  , __float128
#endif // _CCCL_HAS_FLOAT128()
>;
// clang-format on

CUB_TEST(
  "floating_point_to_comparable_int preserves the total order and round-trips", "[util][integer]", CUB_SMALL, fp_types)
{
  using T      = c2h::get<0, TestType>;
  using limits = cuda::std::numeric_limits<T>;
  using cuda::std::__fp_neg;
  const auto zero = cuda::std::__fp_zero<T>();
  const auto one  = cuda::std::__fp_one<T>();
  // clang-format off
  const T values[] = {
    __fp_neg(limits::quiet_NaN()),  // -NaN
    __fp_neg(limits::infinity()),   // -inf
    limits::lowest(),               // -max
    __fp_neg(one),                  // -1.0
    __fp_neg(limits::min()),        // -min
    __fp_neg(limits::denorm_min()), // -denorm_min
    __fp_neg(zero),                 // -0.0
    zero,                           // +0.0
    limits::denorm_min(),           // denorm_min
    limits::min(),                  // min
    one,                            // +1.0
    limits::max(),                  // max
    limits::infinity(),             // +inf
    limits::quiet_NaN()};           // +NaN
  // clang-format on
  for (size_t i = 0; i < sizeof(values) / sizeof(T); ++i)
  {
    const auto int_value = cub::detail::floating_point_to_comparable_int(values[i]);
    const auto fp_value  = cub::detail::comparable_int_to_floating_point<T>(int_value);
    CAPTURE(c2h::type_name<T>(), i);
    REQUIRE(cuda::std::__fp_get_storage(fp_value) == cuda::std::__fp_get_storage(values[i]));
    if (i > 0)
    {
      REQUIRE(cub::detail::floating_point_to_comparable_int(values[i - 1]) < int_value);
    }
  }
}
