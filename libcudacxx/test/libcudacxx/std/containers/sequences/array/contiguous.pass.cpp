//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// An array is a contiguous container

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class C>
__host__ __device__
void test_contiguous ( const C &c )
{
    for ( size_t i = 0; i < c.size(); ++i )
        assert ( *(c.begin() + i) == *(cuda::std::addressof(*c.begin()) + i));
}

int main(int, char**)
{
    {
        typedef double T;
        typedef cuda::std::array<T, 3> C;
        test_contiguous (C());
    }

  return 0;
}
