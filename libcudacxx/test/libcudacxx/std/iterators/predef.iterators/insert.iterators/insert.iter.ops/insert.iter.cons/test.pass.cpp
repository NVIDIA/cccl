//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// insert_iterator

// insert_iterator(Cont& x, Cont::iterator i);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>

#  include "nasty_containers.h"
#  include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::insert_iterator<C> i(c, c.begin());
}

int main(int, char**)
{
  test(cuda::std::vector<int>());
  test(nasty_vector<int>());

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
