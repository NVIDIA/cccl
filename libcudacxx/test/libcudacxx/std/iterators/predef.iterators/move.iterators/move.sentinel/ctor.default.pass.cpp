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

// constexpr move_sentinel();

#include <cuda/std/cassert>
#include <cuda/std/iterator>

__host__ __device__ constexpr bool test()
{
  // The underlying sentinel is an integer.
  {
    cuda::std::move_sentinel<int> m;
    assert(m.base() == 0);
  }

  // The underlying sentinel is a pointer.
  {
    cuda::std::move_sentinel<int*> m;
    assert(m.base() == nullptr);
  }

  // The underlying sentinel is a user-defined type with an explicit default constructor.
  {
    struct S
    {
      constexpr explicit S() = default;
      int i                  = 3;
    };
    cuda::std::move_sentinel<S> m;
    assert(m.base().i == 3);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
