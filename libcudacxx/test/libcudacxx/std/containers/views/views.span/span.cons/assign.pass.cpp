//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

//  constexpr span& operator=(const span& other) noexcept = default;

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/utility>

#include "test_macros.h"

using cuda::std::span;

template <typename T>
__host__ __device__ constexpr bool doAssign(T lhs, T rhs)
{
  static_assert(noexcept(cuda::std::declval<T&>() = rhs));
  lhs = rhs;
  return lhs.data() == rhs.data() && lhs.size() == rhs.size();
}

struct A
{};

__host__ __device__ constexpr bool test()
{
  //  dynamically sized assignment
  {
    int arr[]                    = {5, 6, 7, 9};
    cuda::std::span<int> spans[] = {
      {}, {arr, arr + 1}, {arr, arr + 2}, {arr, arr + 3}, {arr + 1, arr + 3} // same size as s2
    };

    for (size_t i = 0; i < 5; ++i)
    {
      for (size_t j = i; j < 5; ++j)
      {
        assert((doAssign(spans[i], spans[j])));
      }
    }
  }

  //  statically sized assignment
  {
    int arr[]        = {5, 6, 7, 9};
    using spanType   = cuda::std::span<int, 2>;
    spanType spans[] = {spanType{arr, arr + 2}, spanType{arr + 1, arr + 3}, spanType{arr + 2, arr + 4}};

    for (size_t i = 0; i < 3; ++i)
    {
      for (size_t j = i; j < 3; ++j)
      {
        assert((doAssign(spans[i], spans[j])));
      }
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
