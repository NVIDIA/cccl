//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template<class Category, class T, class Distance = ptrdiff_t,
//          class Pointer = T*, class Reference = T&>
// struct iterator
// {
//   typedef T         value_type;
//   typedef Distance  difference_type;
//   typedef Pointer   pointer;
//   typedef Reference reference;
//   typedef Category  iterator_category;
// };

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct A
{};

template <class T>
__host__ __device__ void test2()
{
  typedef cuda::std::iterator<cuda::std::forward_iterator_tag, T> It;
  static_assert((cuda::std::is_same<typename It::value_type, T>::value), "");
  static_assert((cuda::std::is_same<typename It::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<typename It::pointer, T*>::value), "");
  static_assert((cuda::std::is_same<typename It::reference, T&>::value), "");
  static_assert((cuda::std::is_same<typename It::iterator_category, cuda::std::forward_iterator_tag>::value), "");
}

template <class T>
__host__ __device__ void test3()
{
  typedef cuda::std::iterator<cuda::std::bidirectional_iterator_tag, T, short> It;
  static_assert((cuda::std::is_same<typename It::value_type, T>::value), "");
  static_assert((cuda::std::is_same<typename It::difference_type, short>::value), "");
  static_assert((cuda::std::is_same<typename It::pointer, T*>::value), "");
  static_assert((cuda::std::is_same<typename It::reference, T&>::value), "");
  static_assert((cuda::std::is_same<typename It::iterator_category, cuda::std::bidirectional_iterator_tag>::value), "");
}

template <class T>
__host__ __device__ void test4()
{
  typedef cuda::std::iterator<cuda::std::random_access_iterator_tag, T, int, const T*> It;
  static_assert((cuda::std::is_same<typename It::value_type, T>::value), "");
  static_assert((cuda::std::is_same<typename It::difference_type, int>::value), "");
  static_assert((cuda::std::is_same<typename It::pointer, const T*>::value), "");
  static_assert((cuda::std::is_same<typename It::reference, T&>::value), "");
  static_assert((cuda::std::is_same<typename It::iterator_category, cuda::std::random_access_iterator_tag>::value), "");
}

template <class T>
__host__ __device__ void test5()
{
  typedef cuda::std::iterator<cuda::std::input_iterator_tag, T, long, const T*, const T&> It;
  static_assert((cuda::std::is_same<typename It::value_type, T>::value), "");
  static_assert((cuda::std::is_same<typename It::difference_type, long>::value), "");
  static_assert((cuda::std::is_same<typename It::pointer, const T*>::value), "");
  static_assert((cuda::std::is_same<typename It::reference, const T&>::value), "");
  static_assert((cuda::std::is_same<typename It::iterator_category, cuda::std::input_iterator_tag>::value), "");
}

int main(int, char**)
{
  test2<A>();
  test3<A>();
  test4<A>();
  test5<A>();

  return 0;
}
