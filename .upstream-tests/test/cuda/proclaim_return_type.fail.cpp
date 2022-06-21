//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: nvrtc, gcc-4

#include <cuda/functional>

#include <assert.h>

int main(int argc, char ** argv)
{
#ifdef __CUDA_ARCH__
  auto f = cuda::proclaim_return_type<double>(
      [] __device__ () -> int { return 42; });

  assert(f() == 42);
#else
#error shall fail
#endif

  return 0;
}
