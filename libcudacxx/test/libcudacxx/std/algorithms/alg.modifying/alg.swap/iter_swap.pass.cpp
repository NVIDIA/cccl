//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<Iterator Iter1, Iterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   void
//   iter_swap(Iter1 a, Iter2 b);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"

namespace ambiguous
{
struct with_ambiguous_iter_swap
{
  with_ambiguous_iter_swap() = default;
  __host__ __device__ constexpr with_ambiguous_iter_swap(int& val) noexcept
      : val_(cuda::std::addressof(val))
  {}

  __host__ __device__ constexpr int& operator*() noexcept
  {
    return *val_;
  }

  int* val_;
};

template <class Iter1, class Iter2>
_CCCL_API constexpr void iter_swap(Iter1 __a, Iter2 __b)
{
  // do nothing to check whether its preferred
}
} // namespace ambiguous

__host__ __device__ constexpr bool test()
{
  {
    int i = 1;
    int j = 2;
    cuda::std::iter_swap(&i, &j);
    assert(i == 2);
    assert(j == 1);
  }

  {
    int i = 1;
    int j = 2;
    cuda::std::iter_swap(&i, contiguous_iterator<int*>(&j));
    assert(i == 2);
    assert(j == 1);
  }

  {
    int i = 1;
    int j = 2;
    iter_swap(ambiguous::with_ambiguous_iter_swap(i), ambiguous::with_ambiguous_iter_swap(j));

    // Should pick ambiguous::iter_swap
    assert(i == 1);
    assert(j == 2);
  }

  {
    int i = 1;
    int j = 2;
    cuda::std::iter_swap(ambiguous::with_ambiguous_iter_swap(i), ambiguous::with_ambiguous_iter_swap(j));

    // Should pick cuda::std::iter_swap
    assert(i == 2);
    assert(j == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
