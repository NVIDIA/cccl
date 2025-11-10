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
//     typedef bool result_type;

#include <cuda/std/__random_>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    using D           = cuda::std::bernoulli_distribution;
    using result_type = D::result_type;
    static_assert((cuda::std::is_same<result_type, bool>::value), "");
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
