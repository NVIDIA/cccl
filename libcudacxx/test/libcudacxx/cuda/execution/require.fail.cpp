//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__execution/determinism.h>

[[maybe_unused]] _CCCL_GLOBAL_CONSTANT struct query_t
{
} query{};

__host__ __device__ void test()
{
  // not every environment is a requirement
  cuda::std::execution::prop p{query, 42};
  cuda::execution::require(p);
}

int main(int, char**)
{
  test();

  return 0;
}
