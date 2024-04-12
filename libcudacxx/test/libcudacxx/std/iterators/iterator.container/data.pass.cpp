//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <cuda/std/iterator>
// template <class C> constexpr auto data(C& c) -> decltype(c.data());               // C++17
// template <class C> constexpr auto data(const C& c) -> decltype(c.data());         // C++17
// template <class T, size_t N> constexpr T* data(T (&array)[N]) noexcept;           // C++17
// template <class E> constexpr const E* data(initializer_list<E> il) noexcept;      // C++17

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif
#include <cuda/std/array>
#include <cuda/std/initializer_list>

#include "test_macros.h"

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  if TEST_STD_VER > 2014
#    include <cuda/std/string_view>
#  endif
#endif

template <typename C>
__host__ __device__ void test_const_container(const C& c)
{
  //  Can't say noexcept here because the container might not be
  assert(cuda::std::data(c) == c.data());
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c)
{
  ASSERT_NOEXCEPT(cuda::std::data(c));
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
  ASSERT_NOEXCEPT(cuda::std::data(c));
  assert(cuda::std::data(c) == c.begin());
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  ASSERT_NOEXCEPT(cuda::std::data(array));
  assert(cuda::std::data(array) == &array[0]);
}

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int arrA[]{1, 2, 3};

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_VECTOR)
  cuda::std::vector<int> v;
  v.push_back(1);
#endif
  cuda::std::array<int, 1> a;
  a[0]                                = 3;
  cuda::std::initializer_list<int> il = {4};

#if defined(_LIBCUDACXX_HAS_VECTOR)
  test_container(v);
#endif
  test_container(a);
  test_container(il);

#if defined(_LIBCUDACXX_HAS_VECTOR)
  test_const_container(v);
#endif
  test_const_container(a);
  test_const_container(il);

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  if TEST_STD_VER > 2014
  cuda::std::string_view sv{"ABC"};
  test_container(sv);
  test_const_container(sv);
#  endif
#endif

  test_const_array(arrA);

  return 0;
}
