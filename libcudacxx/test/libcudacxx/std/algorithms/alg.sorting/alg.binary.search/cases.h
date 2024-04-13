//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test cases

#ifndef CASES_H
#define CASES_H

#include <cuda/std/array>

#include "test_macros.h"

constexpr size_t num_elements = 1000;

__host__ __device__ TEST_CONSTEXPR_CXX14 cuda::std::array<int, num_elements> get_data(const int M)
{
  cuda::std::array<int, num_elements> arr{};
  cuda::std::size_t i = 0;
  for (int x = 0; x < M; ++x)
  {
    for (size_t j = 0; j < num_elements / M; ++i, ++j)
    {
      arr[i] = x;
    }
  }
  return arr;
}

#endif // CASES_H
