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
  auto t1 = cuda::std::make_pair<int, double>(42, 3.4);
  assert((cuda::std::get<cf>(t1) == cf{1, 2})); // no such type

  return 0;
}
