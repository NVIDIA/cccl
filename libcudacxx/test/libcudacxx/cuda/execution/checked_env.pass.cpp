//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct query1_t
{};

struct query2_t
{};

struct original_query_t
{};

struct custom_env
{
  using property_keys = cuda::execution::property_key_list<original_query_t>;

  TEST_HOST_DEVICE_FUNC constexpr int query(query1_t) const noexcept
  {
    return value;
  }

  TEST_HOST_DEVICE_FUNC constexpr int query(query2_t, int offset) const noexcept
  {
    return value + offset;
  }

  int value;
};

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  using query2        = cuda::execution::property_query<query2_t, int>;
  using expected_keys = cuda::execution::property_key_list<query1_t, query2>;

  auto env = cuda::checked_env<query1_t, query2>(custom_env{42});
  static_assert(cuda::std::is_same_v<typename decltype(env)::property_keys, expected_keys>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(env)>, expected_keys>);
  assert(env.query(query1_t{}) == 42);
  assert(env.query(query2_t{}, 8) == 50);

  custom_env referenced{24};
  auto ref_env = cuda::checked_env<query1_t>(cuda::std::ref(referenced));
  static_assert(
    cuda::std::is_same_v<typename decltype(ref_env)::property_keys, cuda::execution::property_key_list<query1_t>>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(ref_env)>,
                                     cuda::execution::property_key_list<query1_t>>);
  assert(ref_env.query(query1_t{}) == 24);

  return true;
}

int main(int, char**)
{
  test();

#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
