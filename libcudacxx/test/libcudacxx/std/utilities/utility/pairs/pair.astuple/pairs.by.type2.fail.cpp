//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/utility>

int main(int, char**)
{
  typedef cuda::std::complex<float> cf;
  auto t1 = cuda::std::make_pair<int, int>(42, 43);
  assert(cuda::std::get<int>(t1) == 42); // two ints

  return 0;
}
