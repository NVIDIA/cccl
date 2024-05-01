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

// test reset

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    cuda::std::unique_ptr<A> p(new A);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 1);
      assert(B_count == 0);
    }
    A* i = p.get();
    assert(i != nullptr);
    p.reset(new B);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 1);
      assert(B_count == 1);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
  }
  {
    cuda::std::unique_ptr<A> p(new B);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 1);
      assert(B_count == 1);
    }
    A* i = p.get();
    assert(i != nullptr);
    p.reset(new B);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 1);
      assert(B_count == 1);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
    assert(B_count == 0);
  }

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
