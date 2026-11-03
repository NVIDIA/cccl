// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "checked_env_default_queries.h"

using namespace default_query_test;

template <class Env>
TEST_HOST_DEVICE_FUNC constexpr void check_absent(const Env& env)
{
  static_assert(cuda::std::__is_callable_v<get_stop_token_t, const Env&>);
  static_assert(!ex::__queryable_with<const Env&, get_stop_token_t>);
  static_assert(cuda::std::is_same_v<decltype(get_stop_token_t{}(env)), never_stop_token>);
  static_assert(!ex::__detail::__is_valid_property_query_v<Env, cuda::execution::property_query<get_stop_token_t>>);
  assert(!get_stop_token_t{}(env).stop_possible());
  assert(!get_stop_token_t{}(env).stop_requested());

  const auto checked = cuda::checked_env<get_stop_token_t>(env);
  static_assert(
    cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(checked)>, cuda::execution::property_key_list<>>);
  static_assert(!ex::__queryable_with<const decltype(checked)&, get_stop_token_t>);
  assert(!get_stop_token_t{}(checked).stop_possible());
  ex::__detail::__validate_env<decltype(checked)>();

  const provider present{{17, true}};
  const auto first = ex::env{checked, present};
  const auto last  = ex::env{present, checked};
  ex::__detail::__validate_env<decltype(first)>();
  ex::__detail::__validate_env<decltype(last)>();
  assert(get_stop_token_t{}(first).id == 17);
  assert(get_stop_token_t{}(last).id == 17);

  const advertised_provider advertised{{{18, false}}};
  const auto advertised_first = ex::env{checked, advertised};
  const auto advertised_last  = ex::env{advertised, checked};
  ex::__detail::__validate_env<decltype(advertised_first)>();
  ex::__detail::__validate_env<decltype(advertised_last)>();
  assert(get_stop_token_t{}(advertised_first).id == 18);
  assert(get_stop_token_t{}(advertised_last).id == 18);
  assert(get_stop_token_t{}(ex::env{checked, present, checked, advertised}).id == 17);
  assert(get_stop_token_t{}(ex::env{checked, advertised, checked, present}).id == 18);
}

template <class Env>
TEST_HOST_DEVICE_FUNC constexpr void check_provided(const Env& env, int expected_id)
{
  static_assert(ex::__queryable_with<const Env&, get_stop_token_t>);
  static_assert(cuda::std::is_same_v<decltype(get_stop_token_t{}(env)), const provided_stop_token&>);
  const get_stop_token_t get_stop_token{};
  const auto& token = get_stop_token(env);
  assert(token.stop_possible());
  assert(token.id == expected_id);
  assert(token.stop_requested() == ((expected_id & 1) != 0));

  if constexpr (ex::__detail::__has_property_keys<Env>)
  {
    ex::__detail::__validate_env<Env>();
  }
  const auto checked = cuda::checked_env<get_stop_token_t>(env);
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(checked)>,
                                     cuda::execution::property_key_list<get_stop_token_t>>);
  assert(get_stop_token_t{}(checked).id == expected_id);
}

template <class Provider>
TEST_HOST_DEVICE_FUNC constexpr void check_orders(const Provider& value, int expected_id)
{
  check_provided(ex::env{empty_a{}, empty_b{}, value}, expected_id);
  check_provided(ex::env{empty_b{}, empty_a{}, value}, expected_id);
  check_provided(ex::env{empty_a{}, value, empty_b{}}, expected_id);
  check_provided(ex::env{empty_b{}, value, empty_a{}}, expected_id);
  check_provided(ex::env{value, empty_a{}, empty_b{}}, expected_id);
  check_provided(ex::env{value, empty_b{}, empty_a{}}, expected_id);
}

TEST_HOST_DEVICE_FUNC constexpr bool test()
{
  static_assert(cuda::std::__is_callable_v<get_stop_token_t, const int&>);
  const provider first{{1, true}};
  const advertised_provider second{{{2, false}}};
  const auto checked_first  = cuda::checked_env<get_stop_token_t>(first);
  const auto checked_second = cuda::checked_env<get_stop_token_t>(second);
  const auto checked_empty  = cuda::checked_env<>(empty_a{});
  const auto marker         = ex::prop{marker_query_t{}, 42};

  check_absent(empty_a{});
  check_absent(ex::env<>{});
  check_absent(ex::env{empty_a{}});
  check_absent(ex::env{empty_a{}, empty_b{}});
  check_absent(ex::env{empty_a{}, marker});
  check_absent(ex::env{marker, empty_a{}});
  check_absent(ex::env{empty_a{}, empty_b{}, empty_c{}});
  check_absent(checked_empty);
  check_absent(ex::env{checked_empty});
  check_absent(cuda::checked_env<get_stop_token_t>(ex::env<>{}));

  const auto default_only = cuda::checked_env<get_stop_token_t>(marker);
  assert(marker_query_t{}(default_only) == 42);
  assert(marker_query_t{}(ex::env{default_only, ex::prop{marker_query_t{}, 99}}) == 42);

  check_provided(first, 1);
  check_provided(second, 2);
  check_provided(checked_first, 1);
  check_provided(checked_second, 2);
  check_provided(ex::env{first}, 1);
  check_provided(ex::env{second}, 2);
  check_provided(ex::env{empty_a{}, first}, 1);
  check_provided(ex::env{first, empty_a{}}, 1);
  check_provided(ex::env{marker, first}, 1);
  check_provided(ex::env{first, marker}, 1);
  check_provided(ex::env{checked_empty, first}, 1);
  check_provided(ex::env{first, checked_empty}, 1);
  check_provided(ex::env{empty_a{}, checked_first}, 1);
  check_provided(ex::env{checked_first, empty_a{}}, 1);

  check_provided(ex::env{first, second}, 1);
  check_provided(ex::env{second, first}, 2);
  check_provided(ex::env{checked_first, checked_second}, 1);
  check_provided(ex::env{checked_second, checked_first}, 2);
  check_provided(ex::env{empty_a{}, first, empty_b{}, second}, 1);
  check_provided(ex::env{empty_a{}, second, empty_b{}, first}, 2);
  check_orders(first, 1);
  check_orders(second, 2);
  check_orders(checked_first, 1);

  const auto missing_pair = ex::env{empty_a{}, empty_b{}};
  const ex::env<decltype(missing_pair)> nested_missing{missing_pair};
  check_absent(nested_missing);
  check_provided(ex::env{nested_missing, first}, 1);
  check_provided(ex::env{first, nested_missing}, 1);

  const auto present_pair = ex::env{empty_a{}, first};
  const ex::env<decltype(present_pair)> nested_present{present_pair};
  check_provided(nested_present, 1);
  check_provided(ex::env{nested_present, second}, 1);
  check_provided(ex::env{second, nested_present}, 2);

  const auto property = ex::prop{get_stop_token_t{}, provided_stop_token{3, true}};
  check_provided(property, 3);
  check_orders(property, 3);

  // An explicit replacement list validates the resulting interface, not the old list.
  const auto invalid_advertisement = ex::env{advertised_empty_a{}, first};
  const auto overridden            = cuda::checked_env<get_stop_token_t>(invalid_advertisement);
  check_provided(overridden, 1);
  check_provided(ex::env{overridden, empty_a{}}, 1);
  check_absent(cuda::checked_env<>(advertised_empty_a{}));

  int calls = 0;
  const provider counted{{5, true}, &calls};
  const auto counted_env     = ex::env{empty_a{}, counted};
  const auto counted_checked = cuda::checked_env<get_stop_token_t>(counted_env);
  assert(calls == 0);
  assert(get_stop_token_t{}(counted_checked).id == 5);
  assert(calls == 1);

  provider referenced{{7, true}};
  const auto by_reference = cuda::checked_env<get_stop_token_t>(cuda::std::ref(referenced));
  assert(&get_stop_token_t{}(by_reference) == &referenced.token);
  check_orders(by_reference, 7);

  // Only the supplied argument signature is advertised; the other invocation uses its default.
  const auto arguments = cuda::checked_env<rvalue_query, lvalue_query>(argument_provider{});
  static_assert(cuda::std::is_same_v<cuda::execution::property_keys_t<decltype(arguments)>,
                                     cuda::execution::property_key_list<lvalue_query>>);
  int arg = 19;
  assert(&argument_query_t{}(arguments, arg) == &arg);
  assert(!argument_query_t{}(arguments, 0).stop_possible());
  ex::__detail::__validate_env<decltype(arguments)>();

  const auto selected = cuda::checked_env<lvalue_query>(invalid_argument_advertisement{});
  assert(&argument_query_t{}(selected, arg) == &arg);
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
