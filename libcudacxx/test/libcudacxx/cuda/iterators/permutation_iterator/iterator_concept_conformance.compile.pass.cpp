//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Iterator conformance tests for permutation_iterator.

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  static_assert(cuda::std::random_access_iterator<
                cuda::permutation_iterator<contiguous_iterator<int*>, random_access_iterator<int*>>>);

  using Iter = cuda::permutation_iterator<random_access_iterator<int*>, random_access_iterator<int*>>;
  static_assert(cuda::std::indirectly_writable<Iter, int>);
  static_assert(cuda::std::indirectly_swappable<Iter, Iter>);
}

int main(int, char**)
{
  return 0;
}
