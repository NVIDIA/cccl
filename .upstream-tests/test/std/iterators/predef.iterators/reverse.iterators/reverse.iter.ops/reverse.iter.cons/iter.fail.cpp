//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// UNSUPPORTED: nvrtc

// reverse_iterator

// explicit reverse_iterator(Iter x);

// test explicit

#include <cuda/std/iterator>

template <class It>
__host__ __device__
void
test(It i)
{
    cuda::std::reverse_iterator<It> r = i;
}

int main(int, char**)
{
    const char s[] = "123";
    test(s);

  return 0;
}
