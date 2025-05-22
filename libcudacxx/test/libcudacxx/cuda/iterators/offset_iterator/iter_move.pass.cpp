//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// friend constexpr iter_rvalue_reference_t<I>
//   iter_move(const offset_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current)));

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset = 2;
    auto iter        = cuda::offset_iterator(random_access_iterator<int*>{buffer}, offset);
    assert(cuda::std::ranges::iter_move(iter) == buffer[offset]);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(iter)), int&&>);
  }

  {
    const int offset[] = {2};
    auto iter = cuda::offset_iterator(random_access_iterator<int*>{buffer}, random_access_iterator<const int*>{offset});
    assert(cuda::std::ranges::iter_move(iter) == buffer[*offset]);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::iter_move(iter)), int&&>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
