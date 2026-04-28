//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/algorithm.search.h>
#include <cuda/std/cassert>

#include "test_macros.h"

struct equal_to
{
  __host__ __device__ constexpr bool operator()(int a, int b) const
  {
    return a == b;
  }
};

struct searcher
{
  const int *pf, *pl;
  template <class It>
  __host__ __device__ constexpr cuda::std::pair<It, It> operator()(It f, It l) const
  {
    return cuda::std::__search(
      f, l, pf, pl, equal_to{}, cuda::std::random_access_iterator_tag{}, cuda::std::random_access_iterator_tag{});
  }
};

__host__ __device__ constexpr bool test()
{
  constexpr int h[] = {1, 2, 3, 4};
  constexpr int n[] = {2, 3};
  assert(cuda::std::search(h, h + 4, n, n + 2) == h + 1);
  assert(cuda::std::search(h, h + 4, n, n + 2, equal_to{}) == h + 1);

  assert(cuda::std::search(h, h + 4, searcher{n, n + 2}) == h + 1);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
