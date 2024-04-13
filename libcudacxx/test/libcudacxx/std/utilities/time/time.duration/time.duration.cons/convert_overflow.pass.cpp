//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

// overflow should SFINAE instead of error out, LWG 2094

#include <cuda/std/cassert>
#include <cuda/std/chrono>

__host__ __device__ bool f(cuda::std::chrono::milliseconds)
{
  return false;
}

__host__ __device__ bool f(cuda::std::chrono::seconds)
{
  return true;
}

int main(int, char**)
{
  {
    cuda::std::chrono::duration<int, cuda::std::exa> r(1);
    assert(f(r));
  }

  return 0;
}
