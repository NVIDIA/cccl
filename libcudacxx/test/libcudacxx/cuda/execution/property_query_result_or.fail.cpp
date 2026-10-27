// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>

struct query_t
{};

struct invalid_env
{
  using property_keys = cuda::execution::property_key_list<query_t>;

  int query(query_t)
  {
    return 42;
  }
};

int main(int, char**)
{
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  using result = cuda::std::execution::__detail::__advertised_query_result_or_t<invalid_env, query_t, int>;
  return result{}; // expected-error {{use of undeclared identifier 'result'}}
}
