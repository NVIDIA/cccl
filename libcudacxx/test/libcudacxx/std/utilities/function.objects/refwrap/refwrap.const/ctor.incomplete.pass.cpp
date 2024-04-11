//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <functional>
//
// reference_wrapper<T>
//
//  where T is an incomplete type (since C++20)

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct Foo;

__host__ __device__ Foo& get_foo();

__host__ __device__ void test()
{
  Foo& foo = get_foo();
  cuda::std::reference_wrapper<Foo> ref{foo};
  assert(&ref.get() == &foo);
}

struct Foo
{};

__host__ __device__ Foo& get_foo()
{
  static Foo foo;
  return foo;
}

int main(int, char**)
{
  test();
  return 0;
}
