//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N >
// struct array
// {
//     // types:
//     using reference =              T&;
//     using const_reference =        const T&;
//     using iterator =               implementation defined;
//     using const_iterator =         implementation defined;
//     using value_type =             T;
//     using pointer =                T*;
//     using size_type =              size_t;
//     using difference_type =        ptrdiff_t;
//     using value_type =             T;
//     using reverse_iterator =       cuda::std::reverse_iterator<iterator>;
//     using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

#include <cuda/std/array>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class C>
__host__ __device__ void test_iterators()
{
  using ItT  = cuda::std::iterator_traits<typename C::iterator>;
  using CItT = cuda::std::iterator_traits<typename C::const_iterator>;
  static_assert((cuda::std::is_same<typename ItT::iterator_category, cuda::std::random_access_iterator_tag>::value),
                "");
  static_assert((cuda::std::is_same<typename ItT::value_type, typename C::value_type>::value), "");
  static_assert((cuda::std::is_same<typename ItT::reference, typename C::reference>::value), "");
  static_assert((cuda::std::is_same<typename ItT::pointer, typename C::pointer>::value), "");
  static_assert((cuda::std::is_same<typename ItT::difference_type, typename C::difference_type>::value), "");

  static_assert((cuda::std::is_same<typename CItT::iterator_category, cuda::std::random_access_iterator_tag>::value),
                "");
  static_assert((cuda::std::is_same<typename CItT::value_type, typename C::value_type>::value), "");
  static_assert((cuda::std::is_same<typename CItT::reference, typename C::const_reference>::value), "");
  static_assert((cuda::std::is_same<typename CItT::pointer, typename C::const_pointer>::value), "");
  static_assert((cuda::std::is_same<typename CItT::difference_type, typename C::difference_type>::value), "");
}

int main(int, char**)
{
  {
    using T = double;
    using C = cuda::std::array<T, 10>;
    static_assert((cuda::std::is_same<C::reference, T&>::value), "");
    static_assert((cuda::std::is_same<C::const_reference, const T&>::value), "");
    static_assert((cuda::std::is_same<C::iterator, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_iterator, const T*>::value), "");
    test_iterators<C>();
    static_assert((cuda::std::is_same<C::pointer, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_pointer, const T*>::value), "");
    static_assert((cuda::std::is_same<C::size_type, cuda::std::size_t>::value), "");
    static_assert((cuda::std::is_same<C::difference_type, cuda::std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<C::reverse_iterator, cuda::std::reverse_iterator<C::iterator>>::value), "");
    static_assert(
      (cuda::std::is_same<C::const_reverse_iterator, cuda::std::reverse_iterator<C::const_iterator>>::value), "");

    static_assert((cuda::std::is_signed<typename C::difference_type>::value), "");
    static_assert((cuda::std::is_unsigned<typename C::size_type>::value), "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::iterator>::difference_type>::value),
      "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::const_iterator>::difference_type>::value),
      "");
  }
  {
    using T = int*;
    using C = cuda::std::array<T, 0>;
    static_assert((cuda::std::is_same<C::reference, T&>::value), "");
    static_assert((cuda::std::is_same<C::const_reference, const T&>::value), "");
    static_assert((cuda::std::is_same<C::iterator, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_iterator, const T*>::value), "");
    test_iterators<C>();
    static_assert((cuda::std::is_same<C::pointer, T*>::value), "");
    static_assert((cuda::std::is_same<C::const_pointer, const T*>::value), "");
    static_assert((cuda::std::is_same<C::size_type, cuda::std::size_t>::value), "");
    static_assert((cuda::std::is_same<C::difference_type, cuda::std::ptrdiff_t>::value), "");
    static_assert((cuda::std::is_same<C::reverse_iterator, cuda::std::reverse_iterator<C::iterator>>::value), "");
    static_assert(
      (cuda::std::is_same<C::const_reverse_iterator, cuda::std::reverse_iterator<C::const_iterator>>::value), "");

    static_assert((cuda::std::is_signed<typename C::difference_type>::value), "");
    static_assert((cuda::std::is_unsigned<typename C::size_type>::value), "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::iterator>::difference_type>::value),
      "");
    static_assert(
      (cuda::std::is_same<typename C::difference_type,
                          typename cuda::std::iterator_traits<typename C::const_iterator>::difference_type>::value),
      "");
  }

  return 0;
}
