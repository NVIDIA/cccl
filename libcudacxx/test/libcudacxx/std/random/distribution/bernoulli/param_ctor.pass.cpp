//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <random>

// class bernoulli_distribution
// {
//     class param_type;

#include <cuda/std/__random_>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    using D          = cuda::std::bernoulli_distribution;
    using param_type = D::param_type;
    param_type p;
    assert(p.p() == 0.5);
  }
  {
    using D          = cuda::std::bernoulli_distribution;
    using param_type = D::param_type;
    param_type p(0.25);
    assert(p.p() == 0.25);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
