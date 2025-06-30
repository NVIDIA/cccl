//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator[](iter_difference_t<I> n) const;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter  = random_access_iterator<int*>;
  using indexIter = random_access_iterator<const int*>;
  int buffer[]    = {1, 2, 3, 4, 5, 6, 7, 8};

  { // iter::operator[](iter_difference)
    const int offset[] = {5, 2, 7};
    cuda::permutation_iterator iter(baseIter{buffer}, indexIter{offset});
    assert(iter[2] = buffer[offset[2]]);
    static_assert(cuda::std::is_same_v<int&, decltype(iter[0])>);

    // The test iterators are not noexcept
    static_assert(!noexcept(iter[2]));
    static_assert(noexcept(cuda::std::declval<cuda::permutation_iterator<int*, int*>&>()[2]));
  }

  { // const iter::operator[](iter_difference)
    const int offset[] = {5, 2, 7};
    const cuda::permutation_iterator iter(baseIter{buffer}, indexIter{offset});
    assert(iter[2] = buffer[offset[2]]);
    static_assert(cuda::std::is_same_v<int&, decltype(iter[0])>);

    // The test iterators are not noexcept
    static_assert(!noexcept(iter[2]));
    static_assert(noexcept(cuda::std::declval<const cuda::permutation_iterator<int*, int*>&>()[2]));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
