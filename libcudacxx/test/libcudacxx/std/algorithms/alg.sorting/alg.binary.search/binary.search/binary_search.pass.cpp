//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T>
//   constexpr bool      // constexpr after C++17
//   binary_search(Iter first, Iter last, const T& value);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/functional>

#include "../cases.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class T>
__host__ __device__ constexpr void test(Iter first, Iter last, const T& value, bool x)
{
  assert(cuda::std::binary_search(first, last, value) == x);
}

template <class Iter>
__host__ __device__ constexpr void test()
{
  constexpr int M = 10;
  auto v          = get_data(M);
  for (int x = 0; x < M; ++x)
  {
    test(Iter(cuda::std::begin(v)), Iter(cuda::std::end(v)), x, true);
  }
  test(Iter(cuda::std::begin(v)), Iter(cuda::std::end(v)), -1, false);
  test(Iter(cuda::std::begin(v)), Iter(cuda::std::end(v)), M, false);
}

__host__ __device__ constexpr bool test()
{
  test<forward_iterator<const int*>>();
  test<bidirectional_iterator<const int*>>();
  test<random_access_iterator<const int*>>();
  test<const int*>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
