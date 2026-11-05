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

// policy selectors and transformers must be stateless
struct stateful_selector
{
  int state = 0;

  TEST_FUNC constexpr auto operator()(cuda::compute_capability) const -> policy
  {
    return {state};
  }
};

TEST_FUNC void test()
{
  // expected-error@*:* {{Policy selectors must be stateless}}
  // expected-note@*:* 0+ {{}}
  [[maybe_unused]] auto env = cuda::execution::tune(stateful_selector{});
}

int main(int, char**)
{
  test();

  return 0;
}
