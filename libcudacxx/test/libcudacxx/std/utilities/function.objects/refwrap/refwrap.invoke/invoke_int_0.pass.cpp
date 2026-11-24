//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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

// 0 args, return int

TEST_FUNC int f_int_0()
{
  return 3;
}

struct A_int_0
{
  TEST_FUNC int operator()()
  {
    return 4;
  }
};

TEST_FUNC void test_int_0()
{
#if !_CCCL_TILE_COMPILATION() // error: taking address or reference of a function is unsupported in tile mode!
  // function
  {
    cuda::std::reference_wrapper<int()> r1(f_int_0);
    assert(r1() == 3);
  }
#endif // !_CCCL_TILE_COMPILATION()
#if !_CCCL_TILE_COMPILATION() // error: function-to-pointer decay is unsupported in tile code
  // function pointer
  {
    int (*fp)() = f_int_0;
    cuda::std::reference_wrapper<int (*)()> r1(fp);
    assert(r1() == 3);
  }
#endif // !_CCCL_TILE_COMPILATION()
  // functor
  {
    A_int_0 a0;
    cuda::std::reference_wrapper<A_int_0> r1(a0);
    assert(r1() == 4);
  }
}

int main(int, char**)
{
  test_int_0();

  return 0;
}
