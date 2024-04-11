//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// This is for bugs 18853 and 19118

#include <cuda/std/tuple>
// cuda::std::function not supported
// #include <cuda/std/functional>

#include "test_macros.h"

struct X
{
  __device__ __host__ X() {}

  template <class T>
  __device__ __host__ X(T);

  __device__ __host__ void operator()() {}
};

int main(int, char**)
{
  // cuda::std::function not supported
  /*
    X x;
    cuda::std::function<void()> f(x);
  */
  return 0;
}
