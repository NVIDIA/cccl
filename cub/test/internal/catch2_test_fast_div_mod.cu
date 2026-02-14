// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cub/config.cuh>

#include "c2h/catch2_test_helper.h"
#include "c2h/utility.h"

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

using index_types =
  c2h::type_list<int8_t,
                 uint8_t,
                 int16_t,
                 uint16_t,
                 int32_t,
                 uint32_t
#if TEST_INT128()
                 ,
                 int64_t,
                 uint64_t
#endif
                 >;

C2H_TEST("FastDivMod random", "[FastDivMod][Random]", index_types)
{
  using cub::detail::fast_div_mod;
  using index_type         = c2h::get<0, TestType>;
  constexpr auto max_value = +cuda::std::numeric_limits<index_type>::max();
  auto dividend            = GENERATE_COPY(take(20, random(+index_type{1}, max_value)));
  auto divisor             = GENERATE_COPY(take(20, random(+index_type{1}, max_value)));
  fast_div_mod<index_type> div_mod(static_cast<index_type>(divisor));
  CAPTURE(c2h::type_name<index_type>(), dividend, divisor);
  static_assert(std::is_same_v<decltype(dividend / divisor), decltype(div_mod(dividend).quotient)>,
                "quotient type mismatch");
  REQUIRE(dividend / divisor == div_mod(dividend).quotient);
  REQUIRE(dividend % divisor == div_mod(dividend).remainder);
}

C2H_TEST("FastDivMod edge cases", "[FastDivMod][EdgeCases]", index_types)
{
  using cub::detail::fast_div_mod;
  using index_type         = c2h::get<0, TestType>;
  constexpr auto max_value = cuda::std::numeric_limits<index_type>::max();
  CAPTURE(c2h::type_name<index_type>());
  // divisor/dividend == max
  fast_div_mod<index_type> div_mod_max(max_value);
  REQUIRE(1 == div_mod_max(max_value).quotient);
  REQUIRE(0 == div_mod_max(max_value).remainder);
  // divisor == 10, dividend == 0
  fast_div_mod<index_type> div_mod_min(10);
  REQUIRE(0 == div_mod_min(0).quotient);
  REQUIRE(0 == div_mod_min(0).remainder);
}
