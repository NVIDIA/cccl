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
//   front_insert_iterator<Cont>
//   front_inserter(Cont& x);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>

#  include "nasty_containers.h"
#endif // _LIBCUDACXX_HAS_LIST

#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::front_insert_iterator<C> i = cuda::std::front_inserter(c);
  i                                     = 0;
  assert(c.size() == 1);
  assert(c.front() == 0);
}

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
  test(cuda::std::list<int>());
  test(nasty_list<int>());
#endif

  return 0;
}
