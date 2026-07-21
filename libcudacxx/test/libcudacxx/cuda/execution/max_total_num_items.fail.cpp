//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution.max_total_num_items.h>

#include "test_macros.h"

TEST_FUNC void test()
{
  // The bound must be of integral type: a floating-point argument has no viable overload.
  [[maybe_unused]] auto guarantee = cuda::execution::max_total_num_items(1.5);
}

int main(int, char**)
{
  test();

  return 0;
}
