//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: a non-__tile__ variable ("count") cannot be used in tile code

// <functional>

// reference_wrapper

// template <class... ArgTypes>
//   requires Callable<T, ArgTypes&&...>
//   Callable<T, ArgTypes&&...>::result_type
//   operator()(ArgTypes&&... args) const;

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

// 0 args, return void

TEST_GLOBAL_VARIABLE int count = 0;

TEST_FUNC void f_void_0()
{
  ++count;
}

struct A_void_0
{
  TEST_FUNC void operator()()
  {
    ++count;
  }
};

TEST_FUNC void test_void_0()
{
  int save_count = count;
  // function
  {
    cuda::std::reference_wrapper<void()> r1(f_void_0);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    void (*fp)() = f_void_0;
    cuda::std::reference_wrapper<void (*)()> r1(fp);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_void_0 a0;
    cuda::std::reference_wrapper<A_void_0> r1(a0);
    r1();
    assert(count == save_count + 1);
    save_count = count;
  }
}

int main(int, char**)
{
  test_void_0();

  return 0;
}
