//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>
// template <class C> constexpr auto data(C& c) -> decltype(c.data());               // C++17
// template <class C> constexpr auto data(const C& c) -> decltype(c.data());         // C++17
// template <class T, size_t N> constexpr T* data(T (&array)[N]) noexcept;           // C++17
// template <class E> constexpr const E* data(initializer_list<E> il) noexcept;      // C++17

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#include <cuda/std/string_view>

#include "test_macros.h"

template <typename C>
__host__ __device__ void test_const_container(const C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::data(c) == c.data());
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::data(c)));
  assert(cuda::std::data(c) == c.begin());
}

template <typename C>
__host__ __device__ void test_container(C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::data(c) == c.data());
}

template <typename T>
__host__ __device__ void test_container(cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::data(c)));
  assert(cuda::std::data(c) == c.begin());
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  static_assert(noexcept(cuda::std::data(array)));
  assert(cuda::std::data(array) == &array[0]);
}

TEST_GLOBAL_VARIABLE constexpr int arrA[]{1, 2, 3};

int main(int, char**)
{
  cuda::std::inplace_vector<int, 3> v;
  v.push_back(1);
  cuda::std::array<int, 1> a;
  a[0]                                = 3;
  cuda::std::initializer_list<int> il = {4};

  test_container(v);
  test_container(a);
  test_container(il);

  test_const_container(v);
  test_const_container(a);
  test_const_container(il);

  cuda::std::string_view sv{"ABC"};
  test_container(sv);
  test_const_container(sv);

  test_const_array(arrA);

  return 0;
}
