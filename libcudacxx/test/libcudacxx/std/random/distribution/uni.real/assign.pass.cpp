//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>

// template<class RealType = double>
// class uniform_real_distribution

// uniform_real_distribution& operator=(const uniform_real_distribution&);

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ void test()
{
  using D = cuda::std::uniform_real_distribution<float>;
  D d1(2, 5);
  D d2;
  assert(d1 != d2);
  d2 = d1;
  assert(d1 == d2);
}

int main(int, char**)
{
  test();

  return 0;
}
