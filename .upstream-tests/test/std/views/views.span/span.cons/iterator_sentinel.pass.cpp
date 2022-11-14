//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <cuda/std/span>

// template <class It, class End>
// constexpr explicit(Extent != dynamic_extent) span(It first, End last);
// Requires: [first, last) shall be a valid range.
//   If Extent is not equal to dynamic_extent, then last - first shall be equal to Extent.
//

#include <cuda/std/span>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class T, class Sentinel>
__host__ __device__ constexpr bool test_ctor() {
  T val[2] = {};
  auto s1 = cuda::std::span<T>(cuda::std::begin(val), Sentinel(cuda::std::end(val)));
  auto s2 = cuda::std::span<T, 2>(cuda::std::begin(val), Sentinel(cuda::std::end(val)));
  assert(s1.data() == cuda::std::data(val) && s1.size() == cuda::std::size(val));
  assert(s2.data() == cuda::std::data(val) && s2.size() == cuda::std::size(val));
  return true;
}

template <size_t Extent>
__host__ __device__ constexpr void test_constructibility() {
  static_assert(cuda::std::is_constructible_v<cuda::std::span<int, Extent>, int*, int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, const int*, const int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, volatile int*, volatile int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, int*, int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, const int*, const int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<const int, Extent>, volatile int*, volatile int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, int*, int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, const int*, const int*>, "");
  static_assert(cuda::std::is_constructible_v<cuda::std::span<volatile int, Extent>, volatile int*, volatile int*>, "");
  static_assert(!cuda::std::is_constructible_v<cuda::std::span<int, Extent>, int*, float*>, ""); // types wrong
}

__host__ __device__ constexpr bool test() {
  test_constructibility<cuda::std::dynamic_extent>();
  test_constructibility<3>();
  struct A {};
  assert((test_ctor<int, int*>()));
  //assert((test_ctor<int, sized_sentinel<int*>>()));
  assert((test_ctor<A, A*>()));
  //assert((test_ctor<A, sized_sentinel<A*>>()));
  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
