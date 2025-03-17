//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const_reference front() const; // constexpr in C++14
// const_reference back() const;  // constexpr in C++14

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C const c             = {1, 2, 3.5};
    C::const_reference r1 = c.front();
    assert(r1 == 1);

    C::const_reference r2 = c.back();
    assert(r2 == 3.5);
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C const c = {};
    static_assert(cuda::std::is_same_v<decltype(c.back()), C::const_reference>);
    static_assert(noexcept(c.back()));
    static_assert(cuda::std::is_same_v<decltype(c.front()), C::const_reference>);
    static_assert(noexcept(c.front()));
    if (c.size() > (0))
    { // always false
      TEST_IGNORE_NODISCARD c.front();
      TEST_IGNORE_NODISCARD c.back();
    }
  }
  {
    typedef double T;
    typedef cuda::std::array<const T, 0> C;
    C const c = {};
    static_assert(cuda::std::is_same_v<decltype(c.back()), C::const_reference>);
    static_assert(noexcept(c.back()));
    static_assert(cuda::std::is_same_v<decltype(c.front()), C::const_reference>);
    static_assert(noexcept(c.front()));
    if (c.size() > (0))
    {
      TEST_IGNORE_NODISCARD c.front();
      TEST_IGNORE_NODISCARD c.back();
    }
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
