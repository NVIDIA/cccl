
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/std/__random_>

__host__ __device__ void test()
{
  using E = cuda::std::philox4x64;
  E e;
  for (int i = 0; i < 100; ++i)
  {
    e();
  }
}

int main(int, char**)
{
  test();
  return 0;
}
