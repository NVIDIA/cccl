//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator*();
// constexpr decltype(auto) operator*() const
//   requires dereferenceable<const I>;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter = random_access_iterator<int*>;
  int buffer[]   = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    const int offset[] = {5, 3, 1, 7, 5, 2, 6, 1};
    cuda::permutation_iterator iter(baseIter{buffer}, offset);
    for (size_t i = 0; i < 8; ++i, ++iter)
    {
      assert(*iter == buffer[offset[i]]);
    }
  }

  {
    const int offset[] = {2};
    const cuda::permutation_iterator iter(baseIter{buffer}, offset);
    assert(*iter == buffer[*offset]);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
