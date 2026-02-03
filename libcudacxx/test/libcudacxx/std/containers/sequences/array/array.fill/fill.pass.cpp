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

__host__ __device__ constexpr bool tests()
{
  {
    using T = double;
    using C = cuda::std::array<T, 3>;
    C c     = {1, 2, 3.5};
    c.fill(5.5);
    assert(c.size() == 3);
    assert(c[0] == 5.5);
    assert(c[1] == 5.5);
    assert(c[2] == 5.5);
  }

  {
    using T = double;
    using C = cuda::std::array<T, 0>;
    C c     = {};
    c.fill(5.5);
    assert(c.size() == 0);
  }
  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
