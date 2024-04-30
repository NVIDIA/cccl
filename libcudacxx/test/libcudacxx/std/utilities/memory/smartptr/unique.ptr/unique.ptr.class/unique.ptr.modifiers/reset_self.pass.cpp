//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// test reset against resetting self

#include <cuda/std/__memory_>

#include "test_macros.h"

struct A
{
  cuda::std::unique_ptr<A> ptr_;

  __host__ __device__ TEST_CONSTEXPR_CXX23 A()
      : ptr_(this)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 void reset()
  {
    ptr_.reset();
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  (new A)->reset();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
