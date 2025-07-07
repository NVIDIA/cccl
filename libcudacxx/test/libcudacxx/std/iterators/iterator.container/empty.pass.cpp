//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>
// template <class C> constexpr auto empty(const C& c) -> decltype(c.empty());       // C++17
// template <class T, size_t N> constexpr bool empty(const T (&array)[N]) noexcept;  // C++17
// template <class E> constexpr bool empty(initializer_list<E> il) noexcept;         // C++17

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#endif
#include <cuda/std/initializer_list>
#include <cuda/std/string_view>

#include "test_macros.h"

template <typename C>
__host__ __device__ void test_const_container(const C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::empty(c) == c.empty());
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c)
{
  assert(!cuda::std::empty(c));
}

template <typename C>
__host__ __device__ void test_container(C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::empty(c) == c.empty());
}

template <typename T>
__host__ __device__ void test_container(cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::empty(c)));
  assert(!cuda::std::empty(c));
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  static_assert(noexcept(cuda::std::empty(array)));
  assert(!cuda::std::empty(array));
}

TEST_GLOBAL_VARIABLE constexpr int arrA[]{1, 2, 3};

int main(int, char**)
{
  cuda::std::inplace_vector<int, 3> v;
  v.push_back(1);
#if defined(_LIBCUDACXX_HAS_LIST)
  cuda::std::list<int> l;
  l.push_back(2);
#endif
  cuda::std::array<int, 1> a;
  a[0]                                = 3;
  cuda::std::initializer_list<int> il = {4};

  test_container(v);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_container(l);
#endif
  test_container(a);
  test_container(il);

  test_const_container(v);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_const_container(l);
#endif
  test_const_container(a);
  test_const_container(il);

  cuda::std::string_view sv{"ABC"};
  test_container(sv);
  test_const_container(sv);

  test_const_array(arrA);

  return 0;
}
