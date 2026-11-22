// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/execution>

struct member_only_tag
{};

struct member_only_env
{
  int query(member_only_tag) const noexcept
  {
    return 42;
  }
};

int main(int, char**)
{
  // expected-error@* {{The requested query expression is not valid for a const environment.}}
  auto env = cuda::checked_env<member_only_tag>(member_only_env{});
  (void) env;
  return 0;
}
