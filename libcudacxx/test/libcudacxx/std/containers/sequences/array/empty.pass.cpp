//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// class array

// constexpr bool empty() const noexcept;

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    typedef cuda::std::array<int, 2> C;
    C c = {};
    ASSERT_NOEXCEPT(c.empty());
    assert(!c.empty());
  }
  {
    typedef cuda::std::array<int, 0> C;
    C c = {};
    ASSERT_NOEXCEPT(c.empty());
    assert(c.empty());
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif

  // Sanity check for constexpr in C++11
  {
    constexpr cuda::std::array<int, 3> array = {};
    static_assert(!array.empty(), "");
  }

  return 0;
}
