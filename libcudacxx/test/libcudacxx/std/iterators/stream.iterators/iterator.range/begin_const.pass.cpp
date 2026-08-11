//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <class C> auto begin(const C& c) -> decltype(c.begin());

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>

#include "test_macros.h"

int main(int, char**)
{
  int ia[] = {1, 2, 3};
  const cuda::std::inplace_vector<int, 3> v(ia, ia + sizeof(ia) / sizeof(ia[0]));
  cuda::std::inplace_vector<int, 3>::const_iterator i = cuda::std::begin(v);
  assert(*i == 1);

  return 0;
}
