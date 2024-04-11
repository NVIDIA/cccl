//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N> constexpr size_type array<T,N>::size();

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    assert(c.size() == 3);
    assert(c.max_size() == 3);
    assert(!c.empty());
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c = {};
    assert(c.size() == 0);
    assert(c.max_size() == 0);
    assert(c.empty());
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    constexpr C c = {1, 2, 3.5};
    static_assert(c.size() == 3, "");
    static_assert(c.max_size() == 3, "");
    static_assert(!c.empty(), "");
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    constexpr C c = {};
    static_assert(c.size() == 0, "");
    static_assert(c.max_size() == 0, "");
    static_assert(c.empty(), "");
  }

  return 0;
}
