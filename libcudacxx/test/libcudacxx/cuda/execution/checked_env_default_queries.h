// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CHECKED_ENV_DEFAULT_QUERIES_H
#define CHECKED_ENV_DEFAULT_QUERIES_H

#include <cuda/execution>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace default_query_test
{
namespace ex = cuda::std::execution;

struct never_stop_token
{
  TEST_HOST_DEVICE_FUNC constexpr bool stop_possible() const noexcept
  {
    return false;
  }
  TEST_HOST_DEVICE_FUNC constexpr bool stop_requested() const noexcept
  {
    return false;
  }
};

struct provided_stop_token
{
  int id;
  bool requested;

  TEST_HOST_DEVICE_FUNC constexpr bool stop_possible() const noexcept
  {
    return true;
  }
  TEST_HOST_DEVICE_FUNC constexpr bool stop_requested() const noexcept
  {
    return requested;
  }
};

// Like get_stop_token, this CPO has an unconstrained operator and a global default.
struct get_stop_token_t
{
  template <class Env>
  TEST_HOST_DEVICE_FUNC constexpr auto operator()(const Env& env) const noexcept
    -> ex::__query_result_or_t<Env, get_stop_token_t, never_stop_token>
  {
    return ex::__query_or(env, *this, never_stop_token{});
  }
};

// Explicit constructors avoid older Clang crashes on NVCC's output for empty aggregates.
struct empty_a
{
  TEST_HOST_DEVICE_FUNC constexpr empty_a() noexcept {}
};
struct empty_b
{
  TEST_HOST_DEVICE_FUNC constexpr empty_b() noexcept {}
};
struct empty_c
{
  TEST_HOST_DEVICE_FUNC constexpr empty_c() noexcept {}
};

struct provider
{
  provided_stop_token token;
  int* calls = nullptr;

  TEST_HOST_DEVICE_FUNC constexpr const provided_stop_token& query(get_stop_token_t) const noexcept
  {
    if (calls)
    {
      ++*calls;
    }
    return token;
  }
};

struct advertised_provider : provider
{
  using property_keys = cuda::execution::property_key_list<get_stop_token_t>;
};

struct advertised_empty_a
{
  using property_keys = cuda::execution::property_key_list<get_stop_token_t>;
};
struct advertised_empty_b
{
  using property_keys = cuda::execution::property_key_list<get_stop_token_t>;
};

struct marker_query_t : ex::__basic_query<marker_query_t>
{};

struct argument_query_t
{
  template <class Env, class Arg>
  TEST_HOST_DEVICE_FUNC constexpr auto operator()(const Env& env, Arg&& arg) const noexcept
    -> ex::__query_result_or_t<Env, argument_query_t, never_stop_token, Arg>
  {
    return ex::__query_or(env, *this, never_stop_token{}, static_cast<Arg&&>(arg));
  }
};

using lvalue_query = cuda::execution::property_query<argument_query_t, int&>;
using rvalue_query = cuda::execution::property_query<argument_query_t, int>;

struct argument_provider
{
  using property_keys = cuda::execution::property_key_list<lvalue_query>;

  TEST_HOST_DEVICE_FUNC constexpr int& query(argument_query_t, int& arg) const noexcept
  {
    return arg;
  }
};

struct invalid_argument_advertisement : argument_provider
{
  using property_keys = cuda::execution::property_key_list<rvalue_query>;
};
} // namespace default_query_test

#endif // CHECKED_ENV_DEFAULT_QUERIES_H
