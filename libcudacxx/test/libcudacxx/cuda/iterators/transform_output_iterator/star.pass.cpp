//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto operator*() const noexcept(is_nothrow_copy_constructible_v<W>);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

template <class Fn>
__host__ __device__ constexpr void test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  Fn func{};

  {
    cuda::transform_output_iterator iter{buffer, func};
    for (int i = 0; i < 8; ++i, ++iter)
    {
      *iter = i;
      assert(buffer[i] == i + 1);
    }
    static_assert(noexcept(*iter));
    static_assert(noexcept(*iter = 2) == !cuda::std::is_same_v<Fn, PlusOneMayThrow>);
    static_assert(!cuda::std::is_same_v<decltype(*iter), int>);
  }

  {
    const cuda::transform_output_iterator iter{buffer, func};
    *iter = 2;
    assert(buffer[0] == 2 + 1);
    static_assert(noexcept(*iter));
    static_assert(noexcept(*iter = 2) == !cuda::std::is_same_v<Fn, PlusOneMayThrow>);
    static_assert(!cuda::std::is_same_v<decltype(*iter), int>);
  }
}

__host__ __device__ constexpr bool test()
{
  test<PlusOne>();
  test<PlusOneMutable>();
  test<PlusOneMayThrow>();
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test<PlusOneHost>();), (test<PlusOneDevice>();))

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
