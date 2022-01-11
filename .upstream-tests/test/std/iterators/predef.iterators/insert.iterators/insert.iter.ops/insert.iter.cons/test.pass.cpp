//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// insert_iterator

// insert_iterator(Cont& x, Cont::iterator i);

#include <cuda/std/iterator>
#include <cuda/std/vector>
#include "nasty_containers.h"

#include "test_macros.h"

template <class C>
void
test(C c)
{
    cuda::std::insert_iterator<C> i(c, c.begin());
}

int main(int, char**)
{
    test(cuda::std::vector<int>());
    test(nasty_vector<int>());

  return 0;
}
