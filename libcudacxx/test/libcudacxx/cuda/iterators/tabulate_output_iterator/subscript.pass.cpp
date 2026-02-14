//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto operator[](difference_type n) const;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Fn>
__host__ __device__ constexpr void test()
{
  Fn func{};

  {
    cuda::tabulate_output_iterator iter{func, 10};
    for (int i = 10; i < 100; ++i)
    {
      iter[i] = i + 10;
    }
    static_assert(noexcept(iter[10]));
    static_assert(!cuda::std::is_same_v<decltype(iter[10]), void>);
  }

  {
    const cuda::tabulate_output_iterator iter{func, 10};
    for (int i = 10; i < 100; ++i)
    {
      iter[i] = i + 10;
    }
    static_assert(noexcept(iter[10] = 20));
    static_assert(!cuda::std::is_same_v<decltype(iter[10]), void>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<basic_functor>();
  test<mutable_functor>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
