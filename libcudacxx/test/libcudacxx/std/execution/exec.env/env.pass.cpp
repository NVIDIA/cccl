//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: a non-__tile__ variable cannot be used in tile code

#include <cuda/std/execution>

// all other includes follow after <cuda/std/execution>
#include <cuda/std/__type_traits/is_aggregate.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_standard_layout.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_GCC("-Wattributes")

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query1_t
{
} query1{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query2_t
{
} query2{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query3_t
{
} query3{};

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct none_such_t
{
} none_such{};

struct custom_env
{
  TEST_HOST_DEVICE_FUNC constexpr auto query(query1_t) const noexcept
  {
    return -1;
  }

  // A query that takes an extra argument:
  TEST_HOST_DEVICE_FUNC constexpr auto query(query3_t, int i) const noexcept
  {
    return i;
  }
};

struct contextual_env
{
  using property_keys = cuda::execution::property_key_list<cuda::execution::property_query<query3_t, int>>;

  TEST_HOST_DEVICE_FUNC constexpr auto query(query3_t, int i) const noexcept
  {
    return i;
  }
};

struct reserved_and_public_env
{
  using __property_keys = cuda::execution::property_key_list<query1_t>;
  using property_keys   = cuda::execution::property_key_list<query2_t>;
};

struct invalid_reserved_env
{
  using __property_keys = int;
  using property_keys   = cuda::execution::property_key_list<query2_t>;
};

struct invalid_public_env
{
  using __property_keys = cuda::execution::property_key_list<query1_t>;
  using property_keys   = int;
};

struct unmarked_env
{
  using __property_keys = int;
  using property_keys   = double;
};

struct derived_env : cuda::std::execution::env<>
{
  using env::query;

  TEST_HOST_DEVICE_FUNC auto query(query1_t) const
  {
    return 42;
  }
};

template <class Ty>
TEST_HOST_DEVICE_FUNC constexpr bool is_trivial_aggregate()
{
  return cuda::std::is_aggregate_v<Ty> && cuda::std::is_standard_layout_v<Ty> && cuda::std::is_trivially_copyable_v<Ty>
      && cuda::std::is_trivially_constructible_v<Ty> && cuda::std::is_trivially_destructible_v<Ty>;
}

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test()
{
  using cuda::execution::property_key_list;
  using cuda::execution::property_query;

  static_assert(cuda::execution::is_property_key_list_v<property_key_list<>>);
  static_assert(cuda::execution::is_property_key_list_v<property_key_list<query1_t, query2_t>>);
  static_assert(!cuda::execution::is_property_key_list_v<property_query<query1_t>>);

  [[maybe_unused]] cuda::std::execution::env e1{};
  static_assert(cuda::std::is_same_v<decltype(e1), cuda::std::execution::env<>>);
  static_assert(cuda::std::is_same_v<typename decltype(e1)::__property_keys, property_key_list<>>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(e1)>, property_key_list<>>);
  static_assert(is_trivial_aggregate<cuda::std::execution::env<>>());
  static_assert(!cuda::std::execution::__queryable_with<cuda::std::execution::env<>, query1_t>);
  static_assert(sizeof(e1) == 1);

  cuda::std::execution::env e2{cuda::std::execution::prop{query1, 42}};
  assert(e2.query(query1) == 42);
  assert(cuda::std::execution::__query_or(e2, query1, 24) == 42);
  assert(cuda::std::execution::__query_or(e2, query2, 24) == 24);
  static_assert(cuda::std::is_same_v<
                cuda::std::remove_cvref_t<cuda::std::execution::__query_result_or_t<decltype(e2), query1_t, float>>,
                int>);
  static_assert(cuda::std::is_same_v<
                cuda::std::remove_cvref_t<cuda::std::execution::__query_result_or_t<decltype(e2), query2_t, float>>,
                float>);
  using expected_e2_t = cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>>;
  static_assert(cuda::std::is_same_v<decltype(e2), expected_e2_t>);
  static_assert(cuda::std::is_same_v<typename decltype(e2)::__property_keys, property_key_list<query1_t>>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(e2)>, property_key_list<query1_t>>);
  static_assert(is_trivial_aggregate<expected_e2_t>());
  static_assert(cuda::std::is_same_v<decltype(e2.query(query1)), const int&>);
  static_assert(!cuda::std::execution::__queryable_with<expected_e2_t, query2_t>);
  static_assert(sizeof(e2) == sizeof(int));

  cuda::std::execution::env e3{cuda::std::execution::prop{query1, 42}, cuda::std::execution::prop{query2, 3.14}};
  assert(e3.query(query1) == 42);
  assert(e3.query(query2) == 3.14);
  using expected_e3_t =
    cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;
  static_assert(cuda::std::is_same_v<decltype(e3), expected_e3_t>);
  static_assert(cuda::std::is_same_v<typename decltype(e3)::__property_keys, property_key_list<query1_t, query2_t>>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(e3)>, property_key_list<query1_t, query2_t>>);
  static_assert(is_trivial_aggregate<expected_e3_t>());
  static_assert(cuda::std::is_same_v<decltype(e3.query(query1)), const int&>);
  static_assert(cuda::std::is_same_v<decltype(e3.query(query2)), const double&>);

  cuda::std::execution::env e4{
    custom_env{}, cuda::std::execution::prop{query1, 42}, cuda::std::execution::prop{query2, 3.14}};
  assert(e4.query(query1) == -1);
  assert(e4.query(query2) == 3.14);
  assert(e4.query(query3, 42) == 42);
  using expected_e4_t = cuda::std::execution::
    env<custom_env, cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;
  static_assert(cuda::std::is_same_v<decltype(e4), expected_e4_t>);
  static_assert(cuda::std::is_same_v<typename decltype(e4)::__property_keys, property_key_list<query1_t, query2_t>>);
  static_assert(is_trivial_aggregate<expected_e4_t>());
  static_assert(cuda::std::is_same_v<decltype(e4.query(query1)), int>);
  static_assert(cuda::std::is_same_v<decltype(e4.query(query2)), const double&>);
  static_assert(cuda::std::is_same_v<decltype(e4.query(query3, 42)), int>);

  assert(cuda::std::execution::__query_or(e2, query1, 0) == 42);
  assert(cuda::std::execution::__query_or(e2, query2, &e2) == &e2);
  assert(cuda::std::execution::__query_or(e4, query3, 0) == 0);
  assert(cuda::std::execution::__query_or(e4, query3, 0, 42) == 42);

  // Test that env works with const references:
  cuda::std::execution::env<decltype(e2) const&> e5{e2};
  assert(e5.query(query1) == 42);

  cuda::std::execution::env e6{contextual_env{}};
  using contextual_query = property_query<query3_t, int>;
  static_assert(cuda::std::is_same_v<typename decltype(e6)::__property_keys, property_key_list<contextual_query>>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<contextual_env>, property_key_list<contextual_query>>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<const contextual_env&>, property_key_list<contextual_query>>);
  assert(e6.query(query3, 42) == 42);

  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<reserved_and_public_env>, property_key_list<query1_t>>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<const volatile reserved_and_public_env&&>,
                                     property_key_list<query1_t>>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<invalid_reserved_env>, property_key_list<query2_t>>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<invalid_public_env>, property_key_list<query1_t>>);
  static_assert(!cuda::std::execution::__detail::__has_property_keys<unmarked_env>);

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
