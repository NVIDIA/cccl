//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr permutation_iterator()

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using permutation_iterator = cuda::permutation_iterator<random_access_iterator<int*>, random_access_iterator<int*>>;
  permutation_iterator iter;
  assert(iter.base() == random_access_iterator<int*>());

  static_assert(cuda::std::is_copy_constructible_v<permutation_iterator>);
  static_assert(cuda::std::is_move_constructible_v<permutation_iterator>);
  static_assert(cuda::std::is_copy_assignable_v<permutation_iterator>);
  static_assert(cuda::std::is_move_assignable_v<permutation_iterator>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
