//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// move_sentinel

// constexpr explicit move_sentinel(S s);

#include <cuda/std/cassert>
#include <cuda/std/iterator>

__host__ __device__ constexpr bool test()
{
  // The underlying sentinel is an integer.
  {
    static_assert(!cuda::std::is_convertible_v<int, cuda::std::move_sentinel<int>>);
    cuda::std::move_sentinel<int> m(42);
    assert(m.base() == 42);
  }

  // The underlying sentinel is a pointer.
  {
    static_assert(!cuda::std::is_convertible_v<int*, cuda::std::move_sentinel<int*>>);
    int i = 42;
    cuda::std::move_sentinel<int*> m(&i);
    assert(m.base() == &i);
  }

  // The underlying sentinel is a user-defined type with an explicit default constructor.
  {
    struct S
    {
      explicit S() = default;
      __host__ __device__ constexpr explicit S(int j)
          : i(j)
      {}
      int i = 3;
    };
    static_assert(!cuda::std::is_convertible_v<S, cuda::std::move_sentinel<S>>);
    cuda::std::move_sentinel<S> m(S(42));
    assert(m.base().i == 42);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
