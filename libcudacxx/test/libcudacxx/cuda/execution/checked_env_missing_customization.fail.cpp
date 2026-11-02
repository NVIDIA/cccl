// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>
#include <cuda/stream_ref>

struct malformed_env
{
  cuda::stream_ref query(cuda::get_stream_t&) const noexcept
  {
    return cuda::stream_ref{cudaStream_t{}};
  }
};

struct default_value
{
  constexpr int operator()() const noexcept
  {
    return 0;
  }
};

struct default_query_t : cuda::std::execution::__basic_query<default_query_t, default_value>
{};

struct empty_env
{
  using property_keys = cuda::execution::property_key_list<default_query_t>;
};

int main(int, char**)
{
  // A non-const tag reference does not provide the advertised customization.
  // expected-error@* {{The requested query expression is not valid for a const environment.}}
  auto malformed = cuda::checked_env<cuda::get_stream_t>(malformed_env{});
  (void) malformed;

  // A default answer does not provide the advertised member customization either.
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  auto absent = cuda::checked_env<default_query_t>(empty_env{});
  (void) absent;
  return 0;
}
