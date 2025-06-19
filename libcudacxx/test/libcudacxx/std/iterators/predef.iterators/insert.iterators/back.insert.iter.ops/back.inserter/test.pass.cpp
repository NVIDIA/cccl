//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <BackInsertionContainer Cont>
//   back_insert_iterator<Cont>
//   back_inserter(Cont& x);

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>

#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::back_insert_iterator<C> i = cuda::std::back_inserter(c);
  i                                    = 0;
  assert(c.size() == 1);
  assert(c.back() == 0);
}

int main(int, char**)
{
  test(cuda::std::inplace_vector<int, 3>());

  return 0;
}
