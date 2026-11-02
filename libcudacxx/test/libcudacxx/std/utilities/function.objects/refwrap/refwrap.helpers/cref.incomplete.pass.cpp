//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>
//
// reference_wrapper
//
// template <ObjectType T> reference_wrapper<const T> cref(const T& t);
//
//  where T is an incomplete type

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

#if TEST_COMPILER(GCC, >=, 16)
TEST_DIAG_SUPPRESS_GCC("-Wsfinae-incomplete") // defining X, which previously failed to be complete in a SFINAE context
#endif // TEST_COMPILER(GCC, >=, 16)

struct Foo;

TEST_FUNC TEST_FUNC Foo& get_foo();

TEST_FUNC void test()
{
  Foo const& foo                              = get_foo();
  cuda::std::reference_wrapper<Foo const> ref = cuda::std::cref(foo);
  assert(&ref.get() == &foo);
}

struct Foo
{};

TEST_FUNC Foo& get_foo()
{
  static Foo foo;
  return foo;
}

int main(int, char**)
{
  test();
  return 0;
}
