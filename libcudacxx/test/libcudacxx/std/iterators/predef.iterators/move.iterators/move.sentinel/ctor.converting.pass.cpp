//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <cuda/std/iterator>

// move_sentinel

// template<class S2>
//    requires convertible_to<const S2&, S>
//      constexpr move_sentinel(const move_sentinel<S2>& s);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>

struct NonConvertible
{
  __host__ __device__ explicit NonConvertible();
  __host__ __device__ NonConvertible(int i);
  __host__ __device__ explicit NonConvertible(long i) = delete;
};
static_assert(cuda::std::semiregular<NonConvertible>);
static_assert(cuda::std::is_convertible_v<long, NonConvertible>);
static_assert(!cuda::std::convertible_to<long, NonConvertible>);

__host__ __device__ constexpr bool test()
{
  // Constructing from an lvalue.
  {
    cuda::std::move_sentinel<int> m(42);
    cuda::std::move_sentinel<long> m2 = m;
    assert(m2.base() == 42L);
  }

  // Constructing from an rvalue.
  {
    cuda::std::move_sentinel<long> m2 = cuda::std::move_sentinel<int>(43);
    assert(m2.base() == 43L);
  }

  // SFINAE checks.
  {
    static_assert(cuda::std::is_convertible_v<cuda::std::move_sentinel<int>, cuda::std::move_sentinel<long>>);
    static_assert(cuda::std::is_convertible_v<cuda::std::move_sentinel<int*>, cuda::std::move_sentinel<const int*>>);
    static_assert(!cuda::std::is_convertible_v<cuda::std::move_sentinel<const int*>, cuda::std::move_sentinel<int*>>);
    static_assert(cuda::std::is_convertible_v<cuda::std::move_sentinel<int>, cuda::std::move_sentinel<NonConvertible>>);
    static_assert(
      !cuda::std::is_convertible_v<cuda::std::move_sentinel<long>, cuda::std::move_sentinel<NonConvertible>>);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
