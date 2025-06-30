//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr permutation_iterator& operator++();
// constexpr permutation_iterator operator++(int);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter             = random_access_iterator<int*>;
  using indexIter            = random_access_iterator<const int*>;
  using permutation_iterator = cuda::permutation_iterator<baseIter, indexIter>;
  int buffer[]               = {1, 2, 3, 4, 5, 6, 7, 8};
  const int offset[]         = {4, 2, 6};

  const baseIter base{buffer};
  const indexIter off{offset};

  permutation_iterator iter(base, off);
  assert(*iter == buffer[offset[0]]);
  assert(iter++ == permutation_iterator(base, off));
  assert(*iter == buffer[offset[1]]);
  assert(++iter == permutation_iterator(base, off + 2));
  assert(*iter == buffer[offset[2]]);
  assert(iter.index() == offset[2]);

  static_assert(cuda::std::is_same_v<decltype(iter++), permutation_iterator>);
  static_assert(cuda::std::is_same_v<decltype(++iter), permutation_iterator&>);

  // The test iterators are not noexcept
  static_assert(!noexcept(iter++));
  static_assert(!noexcept(++iter));

  // Pointers are noexcept incrementable
  static_assert(noexcept(cuda::permutation_iterator<int*, int*>()++));
  static_assert(noexcept(++cuda::permutation_iterator<int*, int*>()));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
