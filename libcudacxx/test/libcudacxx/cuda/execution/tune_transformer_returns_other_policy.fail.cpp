//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.tune.h>

#include "test_macros.h"

struct policy
{
  int threads_per_block = 1;
};

struct other_policy
{
  int threads_per_block = 1;
};

// a policy transformer must return the policy type it takes as second argument
struct transformer_returns_other_policy
{
  TEST_FUNC constexpr auto operator()(cuda::compute_capability, policy) const -> other_policy
  {
    return {};
  }
};

TEST_FUNC void test()
{
  // expected-error@*:* {{must accept the returned policy type as second argument}}
  // expected-note@*:* 0+ {{}}
  [[maybe_unused]] auto env = cuda::execution::tune(transformer_returns_other_policy{});
}

int main(int, char**)
{
  test();

  return 0;
}
