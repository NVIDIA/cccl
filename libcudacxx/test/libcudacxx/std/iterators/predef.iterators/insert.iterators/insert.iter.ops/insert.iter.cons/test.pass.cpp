//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

// <cuda/std/iterator>

// insert_iterator

// insert_iterator(Cont& x, Cont::iterator i);

#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>

#include "test_macros.h"

template <class C>
TEST_HOST_DEVICE_FUNC void test(C c)
{
  cuda::std::insert_iterator<C> i(c, c.begin());
}

int main(int, char**)
{
  test(cuda::std::inplace_vector<int, 3>());

  return 0;
}
