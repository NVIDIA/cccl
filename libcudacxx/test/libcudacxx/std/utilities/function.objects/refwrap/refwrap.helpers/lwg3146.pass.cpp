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

// template <ObjectType T> reference_wrapper<T> ref(reference_wrapper<T> t);
//   LWG 3146 "Excessive unwrapping in cuda::std::ref/cref"

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::ref(rri);
    static_assert(cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::ref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::cref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  {
    int i = 0;
    cuda::std::reference_wrapper<int> ri(i);
    cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>> rri(ri);
    auto rrj = cuda::std::cref(rri);
    static_assert(
      cuda::std::is_same_v<decltype(rrj), cuda::std::reference_wrapper<const cuda::std::reference_wrapper<int>>>);
    assert(&rrj.get() == &ri);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
