
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

template <typename Engine>
__host__ __device__ void test()
{
  Engine e;
  for (int i = 0; i < 10000; ++i)
  {
    assert(e() <= Engine::max());
  }
  static_assert(Engine::max() > 0, "philox_engine::max() is broken");
}

int main(int, char**)
{
  test<cuda::std::philox4x32>();
  test<cuda::std::philox4x64>();
  return 0;
}
