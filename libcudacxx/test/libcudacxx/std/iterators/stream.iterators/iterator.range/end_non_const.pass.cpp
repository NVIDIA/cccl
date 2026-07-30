//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: a non-__tile__ variable cannot be used in tile code

// <cuda/std/iterator>

// template <class C> auto end(C& c) -> decltype(c.end());

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>

#include "test_macros.h"

int main(int, char**)
{
  int ia[] = {1, 2, 3};
  cuda::std::inplace_vector<int, 3> v(ia, ia + sizeof(ia) / sizeof(ia[0]));
  cuda::std::inplace_vector<int, 3>::iterator i = cuda::std::end(v);
  assert(i == v.end());

  return 0;
}
