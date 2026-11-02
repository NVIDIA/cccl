// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>

struct query_t : cuda::std::execution::__basic_query<query_t>
{};

struct good_env
{
  int query(query_t) const noexcept
  {
    return 42;
  }
};

struct invalid_provider
{
  using property_keys = cuda::execution::property_key_list<query_t>;

  int query(query_t&) const noexcept
  {
    return 17;
  }
};

struct derived_env : cuda::std::execution::env<good_env, invalid_provider>
{};

int main(int, char**)
{
  // The inherited query answers through good_env, but the other advertisement is invalid.
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  cuda::std::execution::__detail::__validate_env<derived_env>();
  return 0;
}
