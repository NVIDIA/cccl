//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__random/feistel_bijection.h>
#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void test()
{
  // Size is nearest power of two >= 256
  auto rng = cuda::std::philox4x64{};
  cuda::__feistel_bijection bijection(100, rng);
  assert(bijection.size() == 256);
  bijection = cuda::__feistel_bijection(256, rng);
  assert(bijection.size() == 256);
  bijection = cuda::__feistel_bijection(257, rng);
  assert(bijection.size() == 512);
  bijection = cuda::__feistel_bijection(1023, rng);
  assert(bijection.size() == 1024);
  bijection = cuda::__feistel_bijection(1024, rng);
  assert(bijection.size() == 1024);
  bijection = cuda::__feistel_bijection(1, rng);
  assert(bijection.size() == 256);
  bijection = cuda::__feistel_bijection(0, rng);
  assert(bijection.size() == 256);
}

int main(int, char**)
{
  test();
  return 0;
}
