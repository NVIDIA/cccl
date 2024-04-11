//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <size_t Extent>
__host__ __device__ constexpr bool test_constructibility()
{
  struct Other
  {};
  static_assert(cuda::std::is_constructible<cuda::std::span<int, Extent>, int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, const int*, size_t>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::span<const int, Extent>, int*, size_t>::value, "");
  static_assert(cuda::std::is_constructible<cuda::std::span<const int, Extent>, const int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, const volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<const int, Extent>, volatile int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<const int, Extent>, const volatile int*, size_t>::value,
                "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<volatile int, Extent>, const int*, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<volatile int, Extent>, const volatile int*, size_t>::value,
                "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, double*, size_t>::value, ""); // iterator
                                                                                                         // type differs
                                                                                                         // from span
                                                                                                         // type
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, size_t, size_t>::value, "");
  static_assert(!cuda::std::is_constructible<cuda::std::span<int, Extent>, Other*, size_t>::value, ""); // unrelated
                                                                                                        // iterator type

  return true;
}

template <class T>
__host__ __device__ constexpr bool test_ctor()
{
  T val[2] = {};
  auto s1  = cuda::std::span<T>(val, 2);
  auto s2  = cuda::std::span<T, 2>(val, 2);
  assert(s1.data() == val && s1.size() == 2);
  assert(s2.data() == val && s2.size() == 2);
  return true;
}

__host__ __device__ constexpr bool test()
{
  test_constructibility<cuda::std::dynamic_extent>();
  test_constructibility<3>();

  struct A
  {};
  test_ctor<int>();
  test_ctor<A>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
