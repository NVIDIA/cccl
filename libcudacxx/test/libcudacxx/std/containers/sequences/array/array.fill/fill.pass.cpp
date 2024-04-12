//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// void fill(const T& u);

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    c.fill(5.5);
    assert(c.size() == 3);
    assert(c[0] == 5.5);
    assert(c[1] == 5.5);
    assert(c[2] == 5.5);
  }

  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c = {};
    c.fill(5.5);
    assert(c.size() == 0);
  }
  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif
  return 0;
}
