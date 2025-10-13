
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/std/random>

template <typename Engine>
__host__ __device__ void test()
{
  Engine e;
  assert(e == e);
  Engine e2;
  assert(e == e2);
  e();
  assert(e != e2);
  e  = Engine(3);
  e2 = Engine(3);
  assert(e == e2);
  e2 = Engine(4);
  assert(e != e2);
}

int main(int, char**)
{
  test<cuda::std::philox4x32>();
  test<cuda::std::philox4x64>();
  return 0;
}
