//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: c++98, c++03

// <cuda/std/iterator>
// template <class C> constexpr auto begin(C& c) -> decltype(c.begin());
// template <class C> constexpr auto begin(const C& c) -> decltype(c.begin());
// template <class C> constexpr auto cbegin(const C& c) -> decltype(cuda::std::begin(c)); // C++14
// template <class C> constexpr auto cend(const C& c) -> decltype(cuda::std::end(c));     // C++14
// template <class C> constexpr auto end  (C& c) -> decltype(c.end());
// template <class C> constexpr auto end  (const C& c) -> decltype(c.end());
// template <class E> constexpr reverse_iterator<const E*> rbegin(initializer_list<E> il);
// template <class E> constexpr reverse_iterator<const E*> rend  (initializer_list<E> il);
//
// template <class C> auto constexpr rbegin(C& c) -> decltype(c.rbegin());                 // C++14
// template <class C> auto constexpr rbegin(const C& c) -> decltype(c.rbegin());           // C++14
// template <class C> auto constexpr rend(C& c) -> decltype(c.rend());                     // C++14
// template <class C> constexpr auto rend(const C& c) -> decltype(c.rend());               // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rbegin(T (&array)[N]);      // C++14
// template <class T, size_t N> reverse_iterator<T*> constexpr rend(T (&array)[N]);        // C++14
// template <class C> constexpr auto crbegin(const C& c) -> decltype(cuda::std::rbegin(c));      // C++14
// template <class C> constexpr auto crend(const C& c) -> decltype(cuda::std::rend(c));          // C++14
//
//  All of these are constexpr in C++17

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#endif
#include <cuda/std/initializer_list>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <iterator>
#  include <utility>
#endif // !TEST_COMPILER(NVRTC)

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
TEST_DIAG_SUPPRESS_GCC("-Wmissing-braces")
TEST_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

template <typename C>
__host__ __device__ void test_const_container(const C& c, typename C::value_type val)
{
  assert(cuda::std::begin(c) == c.begin());
  assert(*cuda::std::begin(c) == val);
  assert(cuda::std::begin(c) != c.end());
  assert(cuda::std::end(c) == c.end());
  assert(cuda::std::cbegin(c) == c.cbegin());
  assert(cuda::std::cbegin(c) != c.cend());
  assert(cuda::std::cend(c) == c.cend());
  assert(cuda::std::rbegin(c) == c.rbegin());
  assert(cuda::std::rbegin(c) != c.rend());
  assert(cuda::std::rend(c) == c.rend());
  assert(cuda::std::crbegin(c) == c.crbegin());
  assert(cuda::std::crbegin(c) != c.crend());
  assert(cuda::std::crend(c) == c.crend());
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c, T val)
{
  assert(cuda::std::begin(c) == c.begin());
  assert(*cuda::std::begin(c) == val);
  assert(cuda::std::begin(c) != c.end());
  assert(cuda::std::end(c) == c.end());

  assert(cuda::std::cbegin(c) != cuda::std::cend(c));
  assert(cuda::std::rbegin(c) != cuda::std::rend(c));
  assert(cuda::std::crbegin(c) != cuda::std::crend(c));
}

template <typename C>
__host__ __device__ void test_container(C& c, typename C::value_type val)
{
  assert(cuda::std::begin(c) == c.begin());
  assert(*cuda::std::begin(c) == val);
  assert(cuda::std::begin(c) != c.end());
  assert(cuda::std::end(c) == c.end());
  assert(cuda::std::cbegin(c) == c.cbegin());
  assert(cuda::std::cbegin(c) != c.cend());
  assert(cuda::std::cend(c) == c.cend());
  assert(cuda::std::rbegin(c) == c.rbegin());
  assert(cuda::std::rbegin(c) != c.rend());
  assert(cuda::std::rend(c) == c.rend());
  assert(cuda::std::crbegin(c) == c.crbegin());
  assert(cuda::std::crbegin(c) != c.crend());
  assert(cuda::std::crend(c) == c.crend());
}

template <typename T>
__host__ __device__ void test_container(cuda::std::initializer_list<T>& c, T val)
{
  assert(cuda::std::begin(c) == c.begin());
  assert(*cuda::std::begin(c) == val);
  assert(cuda::std::begin(c) != c.end());
  assert(cuda::std::end(c) == c.end());
  assert(cuda::std::cbegin(c) != cuda::std::cend(c));
  assert(cuda::std::rbegin(c) != cuda::std::rend(c));
  assert(cuda::std::crbegin(c) != cuda::std::crend(c));
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  assert(cuda::std::begin(array) == array);
  assert(*cuda::std::begin(array) == array[0]);
  assert(cuda::std::begin(array) != cuda::std::end(array));
  assert(cuda::std::end(array) == array + Sz);
  assert(cuda::std::cbegin(array) == array);
  assert(*cuda::std::cbegin(array) == array[0]);
  assert(cuda::std::cbegin(array) != cuda::std::cend(array));
  assert(cuda::std::cend(array) == array + Sz);
}

TEST_GLOBAL_VARIABLE constexpr int global_array[]{1, 2, 3};
TEST_GLOBAL_VARIABLE constexpr int global_const_array[] = {0, 1, 2, 3, 4};

__host__ __device__ void test_ambiguous_std()
{
#if !TEST_COMPILER(NVRTC)
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    {
      cuda::std::array<::std::pair<int, int>, 10> c = {};
      assert(begin(c) == c.begin());
      assert(begin(c) != c.end());
      assert(end(c) == c.end());
    }

    {
      cuda::std::initializer_list<::std::pair<int, int>> init = {{1, 2}};
      assert(begin(init) == init.begin());
      assert(begin(init) != init.end());
      assert(end(init) == init.end());
    }
  ))
  NV_IF_TARGET(NV_IS_HOST, (
    {
      cuda::std::array<::std::pair<int, int>, 10> c = {};
      assert(cbegin(c) == c.cbegin());
      assert(cbegin(c) != c.cend());
      assert(cend(c) == c.cend());
      assert(rbegin(c) == c.rbegin());
      assert(rbegin(c) != c.rend());
      assert(rend(c) == c.rend());
      assert(crbegin(c) == c.crbegin());
      assert(crbegin(c) != c.crend());
      assert(crend(c) == c.crend());
    }

    {
      cuda::std::initializer_list<::std::pair<int, int>> init = {{1, 2}};
      assert(cbegin(init) != cend(init));
      assert(rbegin(init) != rend(init));
      assert(crbegin(init) != crend(init));
    }
  ))

  // clang-format on
#endif // !TEST_COMPILER(NVRTC)
}

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

  test_container(v, 1);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_container(l, 2);
#endif
  test_container(a, 3);
  test_container(il, 4);

  test_const_container(v, 1);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_const_container(l, 2);
#endif
  test_const_container(a, 3);
  test_const_container(il, 4);

  test_const_array(global_array);
  constexpr const int* b = cuda::std::cbegin(global_array);
  constexpr const int* e = cuda::std::cend(global_array);
  static_assert(e - b == 3, "");

  {
    typedef cuda::std::array<int, 5> C;
    constexpr const C local_const_array{0, 1, 2, 3, 4};

    static_assert(local_const_array.begin() == cuda::std::begin(local_const_array), "");
    static_assert(local_const_array.cbegin() == cuda::std::cbegin(local_const_array), "");
    static_assert(local_const_array.end() == cuda::std::end(local_const_array), "");
    static_assert(local_const_array.cend() == cuda::std::cend(local_const_array), "");

    static_assert(local_const_array.rbegin() == cuda::std::rbegin(local_const_array), "");
    static_assert(local_const_array.crbegin() == cuda::std::crbegin(local_const_array), "");
    static_assert(local_const_array.rend() == cuda::std::rend(local_const_array), "");
    static_assert(local_const_array.crend() == cuda::std::crend(local_const_array), "");

    static_assert(cuda::std::begin(local_const_array) != cuda::std::end(local_const_array), "");
    static_assert(cuda::std::rbegin(local_const_array) != cuda::std::rend(local_const_array), "");
    static_assert(cuda::std::cbegin(local_const_array) != cuda::std::cend(local_const_array), "");
    static_assert(cuda::std::crbegin(local_const_array) != cuda::std::crend(local_const_array), "");

    static_assert(*local_const_array.begin() == 0, "");
    static_assert(*local_const_array.rbegin() == 4, "");

    static_assert(*cuda::std::begin(local_const_array) == 0, "");
    static_assert(*cuda::std::cbegin(local_const_array) == 0, "");
    static_assert(*cuda::std::rbegin(local_const_array) == 4, "");
    static_assert(*cuda::std::crbegin(local_const_array) == 4, "");
  }

  {
    static_assert(*cuda::std::begin(global_const_array) == 0, "");
    static_assert(*cuda::std::cbegin(global_const_array) == 0, "");
    static_assert(*cuda::std::rbegin(global_const_array) == 4, "");
    static_assert(*cuda::std::crbegin(global_const_array) == 4, "");
  }

  test_ambiguous_std();

  return 0;
}
