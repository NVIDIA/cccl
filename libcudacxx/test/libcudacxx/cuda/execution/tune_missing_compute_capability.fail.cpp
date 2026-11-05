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

// a policy transformer must also take the compute capability
struct missing_compute_capability
{
  TEST_FUNC constexpr auto operator()(policy p) const -> policy
  {
    return p;
  }
};

TEST_FUNC void test()
{
  // expected-error@*:* {{Policy selectors must be invocable with cuda::compute_capability and an optional policy}}
  // expected-note@*:* 0+ {{}}
  [[maybe_unused]] auto env = cuda::execution::tune(missing_compute_capability{});
}

int main(int, char**)
{
  test();

  return 0;
}
