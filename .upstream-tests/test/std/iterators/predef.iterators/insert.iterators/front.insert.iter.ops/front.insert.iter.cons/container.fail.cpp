//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// front_insert_iterator

// explicit front_insert_iterator(Cont& x);

// test for explicit

#include <cuda/std/iterator>
#include <cuda/std/list>

int main(int, char**)
{
    cuda::std::front_insert_iterator<cuda::std::list<int> > i = cuda::std::list<int>();

  return 0;
}
