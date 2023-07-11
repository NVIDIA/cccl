//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Iterator conformance tests for counted_iterator.

#include <cuda/std/iterator>

#include "test_macros.h"
#include "test_iterators.h"

__host__ __device__ void test() {
  static_assert(cuda::std::input_iterator<cuda::std::counted_iterator<cpp17_input_iterator<int*>>>);
  static_assert(cuda::std::forward_iterator<cuda::std::counted_iterator<forward_iterator<int*>>>);
  static_assert(cuda::std::bidirectional_iterator<cuda::std::counted_iterator<random_access_iterator<int*>>>);
  static_assert(cuda::std::bidirectional_iterator<cuda::std::counted_iterator<contiguous_iterator<int*>>>);
  static_assert(cuda::std::random_access_iterator<cuda::std::counted_iterator<random_access_iterator<int*>>>);
  static_assert(cuda::std::contiguous_iterator<cuda::std::counted_iterator<contiguous_iterator<int*>>>);

  using Iter = cuda::std::counted_iterator<forward_iterator<int*>>;
  static_assert(cuda::std::indirectly_writable<Iter, int>);
  static_assert(cuda::std::indirectly_swappable<Iter, Iter>);
}

int main(int, char**) {
  return 0;
}
