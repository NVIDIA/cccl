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

STATIC_TEST_GLOBAL_VAR int count = 0;

__host__ __device__ int f_int_0()
{
  return 3;
}

struct A_int_0
{
  __host__ __device__ int operator()()
  {
    return 4;
  }
};

__host__ __device__ void test_int_0()
{
  // function
  {
    cuda::std::reference_wrapper<int()> r1(f_int_0);
    assert(r1() == 3);
  }
  // function pointer
  {
    int (*fp)() = f_int_0;
    cuda::std::reference_wrapper<int (*)()> r1(fp);
    assert(r1() == 3);
  }
  // functor
  {
    A_int_0 a0;
    cuda::std::reference_wrapper<A_int_0> r1(a0);
    assert(r1() == 4);
  }
}

// 1 arg, return void

__host__ __device__ void f_void_1(int i)
{
  count += i;
}

struct A_void_1
{
  __host__ __device__ void operator()(int i)
  {
    count += i;
  }
};

int main(int, char**)
{
  test_int_0();

  return 0;
}
