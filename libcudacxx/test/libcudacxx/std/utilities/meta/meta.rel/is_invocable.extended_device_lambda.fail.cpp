//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_invocable should diagnose extended lambdas in host code.

// UNSUPPORTED: clang && (!nvcc)

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class Fn>
void instantiate()
{
  (void) cuda::std::is_invocable_v<Fn>;
}

int main(int, char**)
{
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC)
  instantiate<decltype([] __device__() {})>();
#endif
  return 0;
}
