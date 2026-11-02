//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__functional/call_or.h>
#include <cuda/execution>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct query1_t : cuda::std::execution::__basic_query<query1_t>
{};

struct query2_t : cuda::std::execution::__basic_query<query2_t>
{};

struct query3_t : cuda::std::execution::__basic_query<query3_t>
{};

struct query4_t : cuda::std::execution::__basic_query<query4_t>
{};

struct original_query_t : cuda::std::execution::__basic_query<original_query_t>
{};

struct converting_query_t
{
  template <class Env>
  TEST_HOST_DEVICE_FUNC constexpr long operator()(const Env& env) const noexcept
  {
    return env.query(*this) + 1;
  }
};

struct default_value
{
  TEST_HOST_DEVICE_FUNC constexpr int operator()() const noexcept
  {
    return 0;
  }
};

struct default_query_t : cuda::std::execution::__basic_query<default_query_t, default_value>
{};

struct mutable_env
{
  using property_keys = cuda::execution::property_key_list<query1_t>;

  TEST_HOST_DEVICE_FUNC constexpr int query(query1_t) noexcept
  {
    return 17;
  }
};

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

  TEST_HOST_DEVICE_FUNC constexpr int& query(query3_t, int& arg) const noexcept
  {
    return arg;
  }

  TEST_HOST_DEVICE_FUNC constexpr int query(query4_t, int&& arg) const noexcept
  {
    return value + arg;
  }

  TEST_HOST_DEVICE_FUNC constexpr int query(converting_query_t) const noexcept
  {
    return value;
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
  static_assert(cuda::std::execution::__queryable_with<const decltype(env)&, query4_t, int>);
  assert(env.query(query4_t{}, 8) == 50);

  auto empty_env = cuda::checked_env<>(custom_env{42});
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(empty_env)>, cuda::execution::property_key_list<>>);
  assert(empty_env.query(query1_t{}) == 42);
  assert(empty_env.query(query2_t{}, 8) == 50);

  using query3        = cuda::execution::property_query<query3_t, int&>;
  using query4        = cuda::execution::property_query<query4_t, int>;
  auto forwarding_env = cuda::checked_env<query3, query4>(custom_env{42});
  int arg             = 8;
  static_assert(cuda::std::is_same_v<decltype(forwarding_env.query(query3_t{}, arg)), int&>);
  assert(&forwarding_env.query(query3_t{}, arg) == &arg);
  assert(forwarding_env.query(query4_t{}, 8) == 50);
  static_assert(!cuda::std::execution::__queryable_with<const decltype(forwarding_env)&, original_query_t>);
  static_assert(!cuda::std::execution::__queryable_with<const decltype(forwarding_env)&, query3_t, int>);
  static_assert(!cuda::std::execution::__queryable_with<const decltype(forwarding_env)&, query4_t, int&>);

  custom_env referenced{24};
  auto ref_env = cuda::checked_env<query1_t>(cuda::std::ref(referenced));
  static_assert(
    cuda::std::is_same_v<typename decltype(ref_env)::property_keys, cuda::execution::property_key_list<query1_t>>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(ref_env)>,
                                     cuda::execution::property_key_list<query1_t>>);
  assert(ref_env.query(query1_t{}) == 24);

  const auto converting = cuda::checked_env<converting_query_t>(custom_env{42});
  static_assert(
    cuda::std::is_same_v<cuda::__call_result_or_t<converting_query_t, char, const decltype(converting)&>, long>);
  assert(converting_query_t{}(converting) == 43);

  // A CPO's default answer does not establish that a customization was supplied.
  static_assert(cuda::std::__is_callable_v<default_query_t, const custom_env&>);
  static_assert(
    !cuda::std::execution::__detail::__is_valid_property_query_v<custom_env,
                                                                 cuda::execution::property_query<default_query_t>>);

  // Composition preserves the checked adaptor's replacement list.
  const cuda::std::execution::env composed{env};
  cuda::std::execution::__detail::__validate_env<decltype(composed)>();
  assert(query1_t{}(composed) == 42);

  // Const composition can still refer to a mutable provider through a stored reference.
  mutable_env mutable_source{};
  const cuda::std::execution::env<mutable_env&> referenced_env{mutable_source};
  cuda::std::execution::__detail::__validate_env<decltype(referenced_env)>();
  assert(query1_t{}(referenced_env) == 17);

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
