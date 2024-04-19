//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03
#include <cuda/std/__memory_>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <cuda/std/cassert>

#include "test_macros.h"

//    The only way to create an unique_ptr<T[]> is to default construct them.

class foo
{
public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 foo()
      : val_(3)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX23 int get() const
  {
    return val_;
  }

private:
  int val_;
};

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    auto p1 = cuda::std::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i)
    {
      assert(p1[i] == 0);
    }
  }

#if defined(_LIBCUDACXX_HAS_STRING)
  {
    auto p2 = cuda::std::make_unique<cuda::std::string[]>(5);
    for (int i = 0; i < 5; ++i)
    {
      assert(p2[i].size() == 0);
    }
  }
#endif // _LIBCUDACXX_HAS_STRING

  {
    auto p3 = cuda::std::make_unique<foo[]>(7);
    for (int i = 0; i < 7; ++i)
    {
      assert(p3[i].get() == 3);
    }
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
