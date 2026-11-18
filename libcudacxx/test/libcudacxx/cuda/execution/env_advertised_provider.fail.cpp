// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>
#include <cuda/stream_ref>

struct good_env
{
  cuda::stream_ref query(cuda::get_stream_t) const noexcept
  {
    return cuda::stream_ref{cudaStream_t{}};
  }
};

struct invalid_provider
{
  using property_keys = cuda::execution::property_key_list<cuda::get_stream_t>;

  cuda::stream_ref stream() const noexcept
  {
    return cuda::stream_ref{cudaStream_t{}};
  }

  int query(cuda::get_stream_t) const noexcept
  {
    return 42;
  }
};

int main(int, char**)
{
  const cuda::std::execution::env nested{good_env{}, invalid_provider{}};
  const cuda::std::execution::env env{good_env{}, nested};
  // expected-error@* {{The environment advertises a query expression that is not valid for a const environment.}}
  cuda::std::execution::__detail::__validate_env<decltype(env)>();
  return 0;
}
