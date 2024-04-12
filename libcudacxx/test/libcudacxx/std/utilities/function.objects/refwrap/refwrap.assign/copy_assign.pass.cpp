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

// reference_wrapper& operator=(const reference_wrapper<T>& x);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

class functor1
{};

struct convertible_to_int_ref
{
  int val = 0;
  __host__ __device__ operator int&()
  {
    return val;
  }
  __host__ __device__ operator int const&() const
  {
    return val;
  }
};

template <class T>
__host__ __device__ void test(T& t)
{
  cuda::std::reference_wrapper<T> r(t);
  T t2 = t;
  cuda::std::reference_wrapper<T> r2(t2);
  r2 = r;
  assert(&r2.get() == &t);
}

__host__ __device__ void f() {}
__host__ __device__ void g() {}

__host__ __device__ void test_function()
{
  cuda::std::reference_wrapper<void()> r(f);
  cuda::std::reference_wrapper<void()> r2(g);
  r2 = r;
  assert(&r2.get() == &f);
}

int main(int, char**)
{
  void (*fp)() = f;
  test(fp);
  test_function();
  functor1 f1;
  test(f1);
  int i = 0;
  test(i);
  const int j = 0;
  test(j);

  convertible_to_int_ref convi{};
  test(convi);
  convertible_to_int_ref const convic{};
  test(convic);

  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert((cuda::std::is_assignable<Ref&, int&>::value), "");
    static_assert((!cuda::std::is_assignable<Ref&, int>::value), "");
    static_assert((!cuda::std::is_assignable<Ref&, int&&>::value), "");
  }

  return 0;
}
