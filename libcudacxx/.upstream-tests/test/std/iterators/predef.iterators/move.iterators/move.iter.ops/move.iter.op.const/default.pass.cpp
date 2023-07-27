//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// move_iterator

// move_iterator();
//
//  constexpr in C++17

#include <cuda/std/iterator>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__
void test() {
    cuda::std::move_iterator<It> r;
    unused(r);
}

int main(int, char**) {
  // we don't have a test iterator that is both input and default-constructible, so not testing that case
  test<forward_iterator<char*> >();
  test<bidirectional_iterator<char*> >();
  test<random_access_iterator<char*> >();
  test<char*>();

#if TEST_STD_VER > 14
  {
    constexpr cuda::std::move_iterator<const char *> it;
    unused(it);
  }
#endif

  return 0;
}
