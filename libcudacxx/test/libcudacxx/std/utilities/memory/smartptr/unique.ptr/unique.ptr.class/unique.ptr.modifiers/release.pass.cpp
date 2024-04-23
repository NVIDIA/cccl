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

// test release

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, A[], A>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
  {
    using U = cuda::std::unique_ptr<VT>;
    U u;
    unused(u);
    ASSERT_NOEXCEPT(u.release());
  }
  {
    cuda::std::unique_ptr<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    A* ap = p.get();
    A* a  = p.release();
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == expect_alive);
    }
    assert(p.get() == nullptr);
    assert(ap == a);
    assert(a != nullptr);

    if (IsArray)
    {
      delete[] a;
    }
    else
    {
      delete a;
    }

    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert(A_count == 0);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(A_count == 0);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_basic</*IsArray*/ false>();
  test_basic<true>();

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
