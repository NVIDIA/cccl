//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// This code triggers https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104568
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: icc

// Test the fix for https://llvm.org/PR54100

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"

struct A
{
  int m[0];
};
static_assert(sizeof(A) == 0, ""); // an extension supported by GCC and Clang

int main(int, char**)
{
  {
    cuda::std::unique_ptr<A> p = cuda::std::unique_ptr<A>(new A);
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A[]> p = cuda::std::unique_ptr<A[]>(new A[1]);
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A> p = cuda::std::make_unique<A>();
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A[]> p = cuda::std::make_unique<A[]>(1);
    assert(p != nullptr);
  }

  return 0;
}
