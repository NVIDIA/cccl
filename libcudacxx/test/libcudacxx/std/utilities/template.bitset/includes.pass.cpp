//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test that <cuda/std/bitset> includes <cuda/std/string> and <cuda/std/iosfwd>

#include <cuda/std/bitset>

#include "test_macros.h"

template <class>
__host__ __device__ void test_typedef()
{}

int main(int, char**)
{
#ifdef _LIBCUDACXX_HAS_STRING
  { // test for <cuda/std/string>
    cuda::std::string s;
    ((void) s);
  }
#endif
  { // test for <cuda/std/iosfwd>
    test_typedef<cuda::std::ios>();
    test_typedef<cuda::std::istream>();
    test_typedef<cuda::std::ostream>();
    test_typedef<cuda::std::iostream>();
  }

  return 0;
}
