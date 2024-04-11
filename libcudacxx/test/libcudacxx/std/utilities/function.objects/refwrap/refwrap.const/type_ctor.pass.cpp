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

// reference_wrapper(T& t);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

class functor1
{};

template <class T>
__host__ __device__ void test(T& t)
{
  cuda::std::reference_wrapper<T> r(t);
  assert(&r.get() == &t);
}

__host__ __device__ void f() {}

int main(int, char**)
{
  void (*fp)() = f;
  test(fp);
  test(f);
  functor1 f1;
  test(f1);
  int i = 0;
  test(i);
  const int j = 0;
  test(j);

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_constructible<Ref, int&>::value), "");
    static_assert((!cuda::std::is_constructible<Ref, int>::value), "");
    static_assert((!cuda::std::is_constructible<Ref, int&&>::value), "");
  }

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_nothrow_constructible<Ref, int&>::value), "");
    static_assert((!cuda::std::is_nothrow_constructible<Ref, int>::value), "");
    static_assert((!cuda::std::is_nothrow_constructible<Ref, int&&>::value), "");
  }

  return 0;
}
