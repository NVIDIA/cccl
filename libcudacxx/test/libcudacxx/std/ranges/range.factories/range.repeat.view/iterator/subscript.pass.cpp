//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const W & operator[](difference_type n) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // unbound
  {
    cuda::std::ranges::repeat_view<int> v(31);
    auto iter = v.begin();
    for (size_t i = 0; i < 100; ++i)
    {
      assert(iter[i] == 31);
    }

    static_assert(noexcept(iter[0]));
    static_assert(cuda::std::same_as<decltype(iter[0]), const int&>);
  }

  // bound
  {
    cuda::std::ranges::repeat_view<int, int> v(32);
    auto iter = v.begin();
    for (int i = 0; i < 100; ++i)
    {
      assert(iter[i] == 32);
    }
    static_assert(noexcept(iter[0]));
    static_assert(cuda::std::same_as<decltype(iter[0]), const int&>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
