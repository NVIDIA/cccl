//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<indirectly_swappable<I> I2>
//   friend constexpr void
//     iter_swap(const offset_iterator& x, const offset_iterator<I2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current)));

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto offset_iter1 = cuda::offset_iterator(random_access_iterator<int*>{buffer}, 2);
    auto offset_iter2 = cuda::offset_iterator(random_access_iterator<int*>{buffer}, 4);

    assert(*offset_iter1 == 3);
    assert(*offset_iter2 == 5);
    cuda::std::ranges::iter_swap(offset_iter1, offset_iter2);
    assert(*offset_iter1 == 5);
    assert(*offset_iter2 == 3);
    cuda::std::ranges::iter_swap(offset_iter2, offset_iter1);
    assert(*offset_iter1 == 3);
    assert(*offset_iter2 == 5);
  }

  {
    const int offset1 = 2;
    const int offset2 = 4;
    auto offset_iter1 =
      cuda::offset_iterator(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{&offset1});
    auto offset_iter2 =
      cuda::offset_iterator(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{&offset2});

    assert(*offset_iter1 == 3);
    assert(*offset_iter2 == 5);
    cuda::std::ranges::iter_swap(offset_iter1, offset_iter2);
    assert(*offset_iter1 == 5);
    assert(*offset_iter2 == 3);
    cuda::std::ranges::iter_swap(offset_iter2, offset_iter1);
    assert(*offset_iter1 == 3);
    assert(*offset_iter2 == 5);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
