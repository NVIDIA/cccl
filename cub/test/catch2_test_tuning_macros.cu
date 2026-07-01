// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/tuning_macros.cuh>

#include <cuda/std/type_traits>

#include <format>
#include <sstream>

#include <c2h/catch2_test_helper.h>

CUB_DEFINE_TUNING_POLICY(PolicyInt, (int, int_val))

void test_policy_int()
{
  static_assert(!cuda::std::is_empty_v<cub::PolicyInt>);
  static_assert(cuda::std::is_same_v<int, decltype(cub::PolicyInt::int_val)>);

  static_assert(cuda::std::is_trivially_default_constructible_v<cub::PolicyInt>);
  static_assert(cuda::std::is_trivially_copy_constructible_v<cub::PolicyInt>);

  const cub::PolicyInt p1{10};
  REQUIRE(p1 == p1);
  REQUIRE(!(p1 != p1));

  const cub::PolicyInt p2{3};
  REQUIRE(!(p1 == p2));
  REQUIRE(p1 != p2);

  constexpr auto p1_formatted = "PolicyInt{ .int_val = 10 }";
  std::ostringstream oss{};
  oss << p1;
  REQUIRE(oss.str() == p1_formatted);

#if __cpp_lib_format >= 201907L
  REQUIRE(std::format("{}", p1) == p1_formatted);
#endif // __cpp_lib_format >= 201907L
}

CUB_DEFINE_TUNING_POLICY(PolicyIntEnum, (int, int_val), (BlockStoreAlgorithm, algo))

void test_policy_int_enum()
{
  static_assert(!cuda::std::is_empty_v<cub::PolicyIntEnum>);
  static_assert(cuda::std::is_same_v<int, decltype(cub::PolicyIntEnum::int_val)>);
  static_assert(cuda::std::is_same_v<cub::BlockStoreAlgorithm, decltype(cub::PolicyIntEnum::algo)>);

  static_assert(cuda::std::is_trivially_default_constructible_v<cub::PolicyIntEnum>);
  static_assert(cuda::std::is_trivially_copy_constructible_v<cub::PolicyIntEnum>);

  const cub::PolicyIntEnum p1{10, cub::BLOCK_STORE_DIRECT};
  REQUIRE(p1 == p1);
  REQUIRE(!(p1 != p1));

  const cub::PolicyIntEnum p2{3, cub::BLOCK_STORE_DIRECT};
  REQUIRE(!(p1 == p2));
  REQUIRE(p1 != p2);

  const cub::PolicyIntEnum p3{10, cub::BLOCK_STORE_STRIPED};
  REQUIRE(!(p1 == p3));
  REQUIRE(p1 != p3);

  constexpr auto p1_formatted = "PolicyIntEnum{ .int_val = 10, .algo = BLOCK_STORE_DIRECT }";
  std::ostringstream oss{};
  oss << p1;
  REQUIRE(oss.str() == p1_formatted);

#if __cpp_lib_format >= 201907L
  REQUIRE(std::format("{}", p1) == p1_formatted);
#endif // __cpp_lib_format >= 201907L
}

C2H_TEST("Tuning macros", "")
{
  test_policy_int();
  test_policy_int_enum();
}
