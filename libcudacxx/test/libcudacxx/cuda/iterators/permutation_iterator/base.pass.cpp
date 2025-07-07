//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr const I& base() const &;
// constexpr I base() &&;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using baseIter = random_access_iterator<int*>;
  int buffer[]   = {1, 2, 3, 4, 5, 6, 7, 8};

  const int offset[] = {2};
  cuda::permutation_iterator iter(baseIter{buffer}, offset);
  assert(base(iter.base()) == buffer);
  assert(base(cuda::std::move(iter).base()) == buffer);

  static_assert(cuda::std::is_same_v<decltype(iter.base()), const baseIter&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::move(iter).base()), baseIter>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
