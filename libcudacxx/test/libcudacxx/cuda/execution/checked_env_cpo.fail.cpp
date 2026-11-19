// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>
#include <cuda/stream_ref>

struct invalid_env
{
  int query(cuda::get_stream_t) const noexcept
  {
    return 42;
  }
};

int main(int, char**)
{
  // expected-error@* {{The requested query expression is not valid for a const environment.}}
  auto env = cuda::checked_env<cuda::get_stream_t>(invalid_env{});
  (void) env;
  return 0;
}
