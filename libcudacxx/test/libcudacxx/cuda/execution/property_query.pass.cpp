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
#include <cuda/std/type_traits>

#include "test_macros.h"

struct query1_t
{};

struct query2_t
{};

struct query3_t
{};

struct custom_env
{};

template <class... Queries>
struct advertised_env
{
  using property_keys = cuda::execution::property_key_list<Queries...>;
};

struct contextual_env
{
  using property_keys = cuda::execution::property_key_list<cuda::execution::property_query<query3_t, int>>;

  TEST_HOST_DEVICE_FUNC constexpr int query(query3_t, int value) const noexcept
  {
    return value;
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

template <class T, class = void>
inline constexpr bool has_property_keys = false;

template <class T>
inline constexpr bool has_property_keys<T, cuda::std::void_t<cuda::execution::property_keys_t<T>>> = true;

template <class List, class... Queries>
inline constexpr bool has_exact_property_queries = false;

template <class... Actual, class... Queries>
inline constexpr bool has_exact_property_queries<cuda::execution::property_key_list<Actual...>, Queries...> =
  sizeof...(Actual) == sizeof...(Queries) && (cuda::std::__is_one_of_v<Queries, Actual...> && ...);

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  using empty_list       = cuda::execution::property_key_list<>;
  using query1_list      = cuda::execution::property_key_list<query1_t>;
  using query2_list      = cuda::execution::property_key_list<query2_t>;
  using query12_list     = cuda::execution::property_key_list<query1_t, query2_t>;
  using contextual_query = cuda::execution::property_query<query3_t, int>;
  using contextual_list  = cuda::execution::property_key_list<contextual_query>;
  using query1           = cuda::execution::property_query<query1_t>;
  using query2           = cuda::execution::property_query<query2_t>;

  static_assert(cuda::execution::is_property_key_list_v<empty_list>);
  static_assert(cuda::execution::is_property_key_list_v<query12_list>);
  static_assert(!cuda::execution::is_property_key_list_v<contextual_query>);

  using env0 = cuda::std::execution::env<>;
  using env1 = cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>>;
  using env2 =
    cuda::std::execution::env<cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;
  using env3 = cuda::std::execution::
    env<custom_env, cuda::std::execution::prop<query1_t, int>, cuda::std::execution::prop<query2_t, double>>;

  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<env0>, empty_list>);
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<env1>, query1>);
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<env2>, query1, query2>);
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<env3>, query1, query2>);

  using only_unmarked = cuda::std::execution::env<custom_env, custom_env, custom_env>;
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<only_unmarked>, empty_list>);

  using duplicates = cuda::std::execution::env<advertised_env<query1_t, query1, query2_t, query2, query1_t>>;
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<duplicates>, query1, query2>);

  using repeated_components = cuda::std::execution::env<env1, duplicates, env2, custom_env>;
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<repeated_components>, query1, query2>);

  using contextual_ref_query  = cuda::execution::property_query<query3_t, int&>;
  using contextual_rref_query = cuda::execution::property_query<query3_t, int&&>;
  using contextual_duplicates = cuda::std::execution::
    env<contextual_env, advertised_env<contextual_query, contextual_ref_query, contextual_rref_query, contextual_query>>;
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<contextual_duplicates>,
                                           contextual_query,
                                           contextual_ref_query,
                                           contextual_rref_query>);
  static_assert(cuda::std::execution::__detail::__advertises_query_v<contextual_duplicates, query3_t, int>);
  static_assert(cuda::std::execution::__detail::__advertises_query_v<contextual_duplicates, query3_t, int&>);
  static_assert(cuda::std::execution::__detail::__advertises_query_v<contextual_duplicates, query3_t, int&&>);
  static_assert(!cuda::std::execution::__detail::__advertises_query_v<contextual_duplicates, query3_t>);
  static_assert(!cuda::std::execution::__detail::__advertises_query_v<contextual_duplicates, query3_t, double>);
  static_assert(cuda::std::execution::__detail::__advertises_query_v<duplicates, query1_t>);
  static_assert(!cuda::std::execution::__detail::__advertises_query_v<duplicates, query1_t, int>);
  static_assert(!cuda::std::execution::__detail::__advertises_query_v<env0, query1_t>);

  using prop1  = cuda::std::execution::prop<query1_t, int>;
  auto wrapped = cuda::checked_env<query1>(prop1{query1_t{}, 22});
  cuda::std::execution::env<prop1, decltype(wrapped)> first{prop1{query1_t{}, 11}, wrapped};
  cuda::std::execution::env<decltype(wrapped), prop1> second{wrapped, prop1{query1_t{}, 11}};
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<decltype(first)>, query1>);
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<decltype(second)>, query1>);
  assert(first.query(query1_t{}) == 11);
  assert(second.query(query1_t{}) == 22);
  cuda::std::execution::env<decltype(first), prop1> nested{first, prop1{query1_t{}, 33}};
  static_assert(has_exact_property_queries<cuda::execution::property_keys_t<decltype(nested)>, query1>);
  assert(nested.query(query1_t{}) == 11);

  cuda::std::execution::env contextual{contextual_env{}};
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(contextual)>, contextual_list>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<contextual_env>, contextual_list>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<const contextual_env&>, contextual_list>);
  assert(contextual.query(query3_t{}, 42) == 42);

  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<reserved_and_public_env>, query1_list>);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<const volatile reserved_and_public_env&&>, query1_list>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<invalid_reserved_env>, query2_list>);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<invalid_public_env>, query1_list>);
  static_assert(!has_property_keys<unmarked_env>);

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
