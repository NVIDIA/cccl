//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/memory>

// template <ObjectType T> T* addressof(T& r);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

#ifdef TEST_COMPILER_CLANG_CUDA
#  include <new>
#endif // TEST_COMPILER_CLANG_CUDA

struct A
{
  __host__ __device__ void operator&() const {}
};

struct nothing
{
  __host__ __device__ operator char&()
  {
    static char c;
    return c;
  }
};

int main(int, char**)
{
  {
    int i;
    double d;
    assert(cuda::std::addressof(i) == &i);
    assert(cuda::std::addressof(d) == &d);
    A* tp        = new A;
    const A* ctp = tp;
    assert(cuda::std::addressof(*tp) == tp);
    assert(cuda::std::addressof(*ctp) == tp);
    delete tp;
  }
  {
    union
    {
      nothing n;
      int i;
    };
    assert(cuda::std::addressof(n) == (void*) cuda::std::addressof(i));
  }

  return 0;
}
