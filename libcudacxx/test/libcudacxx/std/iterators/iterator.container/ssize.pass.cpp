//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>
// template <class C> constexpr auto ssize(const C& c)
//     -> common_type_t<ptrdiff_t, make_signed_t<decltype(c.size())>>;                    // C++20
// template <class T, ptrdiff_t> constexpr ptrdiff_t ssize(const T (&array)[N]) noexcept; // C++20

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

TEST_DIAG_SUPPRESS_GCC("-Wtype-limits")

struct short_container
{
  __host__ __device__ uint16_t size() const
  {
    return 60000;
  } // not noexcept
};

template <typename C>
__host__ __device__ void test_container(C& c)
{
  //  Can't say noexcept here because the container might not be
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(c))>, "");
  assert(cuda::std::ssize(c) == static_cast<decltype(cuda::std::ssize(c))>(c.size()));
}

template <typename C>
__host__ __device__ void test_const_container(const C& c)
{
  //  Can't say noexcept here because the container might not be
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(c))>, "");
  assert(cuda::std::ssize(c) == static_cast<decltype(cuda::std::ssize(c))>(c.size()));
}

template <typename T>
__host__ __device__ void test_const_container(const cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::ssize(c))); // our cuda::std::ssize is conditionally noexcept
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(c))>, "");
  assert(cuda::std::ssize(c) == static_cast<decltype(cuda::std::ssize(c))>(c.size()));
}

template <typename T>
__host__ __device__ void test_container(cuda::std::initializer_list<T>& c)
{
  static_assert(noexcept(cuda::std::ssize(c))); // our cuda::std::ssize is conditionally noexcept
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(c))>, "");
  assert(cuda::std::ssize(c) == static_cast<decltype(cuda::std::ssize(c))>(c.size()));
}

template <typename T, size_t Sz>
__host__ __device__ void test_const_array(const T (&array)[Sz])
{
  static_assert(noexcept(cuda::std::ssize(array)));
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(array))>, "");
  assert(cuda::std::ssize(array) == Sz);
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
  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(v))>);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_container(l);
  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(l))>);
#endif
  test_container(a);
  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(a))>);
  test_container(il);
  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(il))>);

  test_const_container(v);
#if defined(_LIBCUDACXX_HAS_LIST)
  test_const_container(l);
#endif
  test_const_container(a);
  test_const_container(il);

  cuda::std::string_view sv{"ABC"};
  test_container(sv);
  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(sv))>);
  test_const_container(sv);

  static_assert(cuda::std::is_same_v<ptrdiff_t, decltype(cuda::std::ssize(arrA))>);
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(arrA))>, "");
  test_const_array(arrA);

  //  From P1227R2:
  //     Note that the code does not just return the cuda::std::make_signed variant of
  //     the container's size() method, because it's conceivable that a container
  //     might choose to represent its size as a uint16_t, supporting up to
  //     65,535 elements, and it would be a disaster for cuda::std::ssize() to turn a
  //     size of 60,000 into a size of -5,536.

  short_container sc;
  //  is the return type signed? Is it big enough to hold 60K?
  //  is the "signed version" of sc.size() too small?
  static_assert(cuda::std::is_signed_v<decltype(cuda::std::ssize(sc))>, "");
  static_assert(cuda::std::numeric_limits<decltype(cuda::std::ssize(sc))>::max() > 60000, "");
  static_assert(cuda::std::numeric_limits<cuda::std::make_signed_t<decltype(cuda::std::size(sc))>>::max() < 60000, "");
  NV_IF_TARGET(NV_IS_DEVICE, (assert(cuda::std::ssize(sc) == 60000);))
  static_assert(!noexcept(cuda::std::ssize(sc)));

  return 0;
}
