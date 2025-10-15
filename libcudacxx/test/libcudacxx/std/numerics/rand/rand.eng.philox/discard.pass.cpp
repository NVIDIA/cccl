
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

#include <cuda/std/__random/philox_engine.h>

template <typename Engine, typename Engine::result_type value_10000>
__host__ __device__ void test()
{
  Engine e;
  for (int i = 0; i < 100; ++i)
  {
    Engine e2;
    e2.discard(i);
    assert(e == e2);
    e();
  }

  e = Engine();
  e.discard(9999);
  assert(e() == value_10000);
}

int main(int, char**)
{
  test<cuda::std::philox4x32, 1955073260u>();
  test<cuda::std::philox4x64, 3409172418970261260ull>();

  return 0;
}
