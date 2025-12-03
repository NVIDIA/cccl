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

// template <ObjectType T> reference_wrapper<const T> cref(reference_wrapper<T> t);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

namespace adl
{
struct A
{};
__host__ __device__ void cref(A) {}
} // namespace adl

__host__ __device__ constexpr bool test()
{
  {
    const int i                                = 0;
    cuda::std::reference_wrapper<const int> r1 = cuda::std::cref(i);
    cuda::std::reference_wrapper<const int> r2 = cuda::std::cref(r1);
    assert(&r2.get() == &i);
  }
  {
    adl::A a{};
    cuda::std::reference_wrapper<const adl::A> a1 = cuda::std::cref(a);
    cuda::std::reference_wrapper<const adl::A> a2 = cuda::std::cref(a1);
    assert(&a2.get() == &a);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
