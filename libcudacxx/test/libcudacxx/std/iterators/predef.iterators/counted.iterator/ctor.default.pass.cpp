//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr counted_iterator() requires default_initializable<I> = default;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  static_assert(!cuda::std::default_initializable<cuda::std::counted_iterator<cpp17_input_iterator<int*>>>);
  static_assert(cuda::std::default_initializable<cuda::std::counted_iterator<forward_iterator<int*>>>);

  cuda::std::counted_iterator<forward_iterator<int*>> iter;
  assert(iter.base() == forward_iterator<int*>());
  assert(iter.count() == 0);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
