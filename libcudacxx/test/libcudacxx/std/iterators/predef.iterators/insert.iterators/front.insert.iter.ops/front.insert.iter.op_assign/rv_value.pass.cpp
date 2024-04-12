//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(Cont::value_type&& value);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#  include <cuda/std/memory>
#endif
#include <cuda/std/cassert>

#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::front_insert_iterator<C> i(c);
  i = typename C::value_type();
  assert(c.front() == typename C::value_type());
}

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
  test(cuda::std::list<cuda::std::unique_ptr<int>>());
#endif

  return 0;
}
