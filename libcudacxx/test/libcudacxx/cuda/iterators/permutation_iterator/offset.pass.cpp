//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iter_difference_t<I> offset() const noexcept;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset[] = {2};
    cuda::permutation_iterator iter(buffer, random_access_iterator<const int*>{offset});
    assert(iter.offset() == random_access_iterator<const int*>{offset});
    assert(cuda::std::move(iter).offset() == random_access_iterator<const int*>{offset});
    static_assert(cuda::std::is_same_v<decltype(iter.offset()), const random_access_iterator<const int*>&>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).offset()), random_access_iterator<const int*>>);
    static_assert(noexcept(iter.offset()));
    static_assert(
      noexcept(cuda::std::move(iter).offset()) == cuda::std::is_move_constructible_v<random_access_iterator<int*>>);
  }

  {
    const int offset[] = {2};
    const cuda::permutation_iterator iter(buffer, random_access_iterator<const int*>{offset});
    assert(iter.offset() == random_access_iterator<const int*>{offset});
    static_assert(cuda::std::is_same_v<decltype(iter.offset()), const random_access_iterator<const int*>&>);
    static_assert(noexcept(iter.offset()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
