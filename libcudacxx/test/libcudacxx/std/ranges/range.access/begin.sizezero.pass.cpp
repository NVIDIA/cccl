//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc

// cuda::std::ranges::begin
// cuda::std::ranges::cbegin
//   Test the fix for https://llvm.org/PR54100

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_macros.h"

#ifndef __CUDA_ARCH__
struct A
{
  int m[0];
};
static_assert(sizeof(A) == 0); // an extension supported by GCC and Clang

__device__ static A a[10];

int main(int, char**)
{
  auto p = cuda::std::ranges::begin(a);
  static_assert(cuda::std::same_as<A*, decltype(cuda::std::ranges::begin(a))>);
  assert(p == a);
  auto cp = cuda::std::ranges::cbegin(a);
  static_assert(cuda::std::same_as<const A*, decltype(cuda::std::ranges::cbegin(a))>);
  assert(cp == a);

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
