// SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include "cub_test_macros.h"
#include "test_util.h"

template <typename T>
std::string print(T val)
{
  std::stringstream ss;
  ss << val;
  return ss.str();
}

#if TEST_INT128()
CUB_TEST_CASE("Test utils can print __int128", "[test][utils]", CUB_SMALL)
{
  REQUIRE(print(__int128_t{0}) == "0");
  REQUIRE(print(__int128_t{42}) == "42");
  REQUIRE(print(__int128_t{-1}) == "-1");
  REQUIRE(print(__int128_t{-42}) == "-42");
  REQUIRE(print(-1 * (__int128_t{1} << 120)) == "-1329227995784915872903807060280344576");
}

CUB_TEST_CASE("Test utils can print __uint128", "[test][utils]", CUB_SMALL)
{
  REQUIRE(print(__uint128_t{0}) == "0");
  REQUIRE(print(__uint128_t{1}) == "1");
  REQUIRE(print(__uint128_t{42}) == "42");
  REQUIRE(print(__uint128_t{1} << 120) == "1329227995784915872903807060280344576");
}
#endif

CUB_TEST_CASE("Test utils can print KeyValuePair", "[test][utils]", CUB_SMALL)
{
  REQUIRE(print(cub::KeyValuePair<int, int>{42, -42}) == "(42,-42)");
}
