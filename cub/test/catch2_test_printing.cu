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

CUB_TEST_CASE("Catch2 can stringify 128-bit integers", "[test][utils]", CUB_SMALL)
{
  const __int128_t signed_value    = -42;
  const __int128_t signed_expected = -42;
  REQUIRE(signed_value == signed_expected);
  REQUIRE(Catch::StringMaker<__int128_t>::convert(signed_value) == "-42");

  const __uint128_t unsigned_value    = __uint128_t{1} << 120;
  const __uint128_t unsigned_expected = __uint128_t{1} << 120;
  REQUIRE(unsigned_value == unsigned_expected);
  REQUIRE(Catch::StringMaker<__uint128_t>::convert(unsigned_value) == "1329227995784915872903807060280344576");
}
#endif

CUB_TEST_CASE("Test utils can print KeyValuePair", "[test][utils]", CUB_SMALL)
{
  REQUIRE(print(cub::KeyValuePair<int, int>{42, -42}) == "(42,-42)");
}
