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

// a callable must either be a policy selector or a policy transformer, but not both
struct selector_and_transformer
{
  TEST_FUNC constexpr auto operator()(cuda::compute_capability) const -> policy
  {
    return {};
  }

  TEST_FUNC constexpr auto operator()(cuda::compute_capability, policy p) const -> policy
  {
    return p;
  }
};

TEST_FUNC void test()
{
  [[maybe_unused]] auto env = cuda::execution::tune(selector_and_transformer{});
}

int main(int, char**)
{
  test();

  return 0;
}
