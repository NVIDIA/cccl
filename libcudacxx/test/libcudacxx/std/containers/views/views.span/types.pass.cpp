//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// template<class ElementType, size_t Extent = dynamic_extent>
// class span {
// public:
//  // constants and types
//  using element_type           = ElementType;
//  using value_type             = remove_cv_t<ElementType>;
//  using size_type              = size_t;
//  using difference_type        = ptrdiff_t;
//  using pointer                = element_type *;
//  using reference              = element_type &;
//  using const_pointe           = const element_type *;
//  using const_reference        = const element_type &;
//  using iterator               = implementation-defined;
//  using reverse_iterator       = std::reverse_iterator<iterator>;
//
//  static constexpr size_type extent = Extent;
//

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/span>

#include "test_macros.h"

template <typename S, typename Iter>
__host__ __device__ void testIterator()
{
  typedef cuda::std::iterator_traits<Iter> ItT;

  static_assert(cuda::std::is_same_v<typename ItT::iterator_category, cuda::std::random_access_iterator_tag>);
  static_assert(cuda::std::is_same_v<typename ItT::value_type, typename S::value_type>);
  static_assert(cuda::std::is_same_v<typename ItT::reference, typename S::reference>);
  static_assert(cuda::std::is_same_v<typename ItT::pointer, typename S::pointer>);
  static_assert(cuda::std::is_same_v<typename ItT::difference_type, typename S::difference_type>);
}

template <typename S, typename ElementType, cuda::std::size_t Size>
__host__ __device__ void testSpan()
{
  static_assert(cuda::std::is_same_v<typename S::element_type, ElementType>);
  static_assert(cuda::std::is_same_v<typename S::value_type, typename cuda::std::remove_cv<ElementType>::type>);
  static_assert(cuda::std::is_same_v<typename S::size_type, cuda::std::size_t>);
  static_assert(cuda::std::is_same_v<typename S::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::is_same_v<typename S::pointer, ElementType*>);
  static_assert(cuda::std::is_same_v<typename S::const_pointer, const ElementType*>);
  static_assert(cuda::std::is_same_v<typename S::reference, ElementType&>);
  static_assert(cuda::std::is_same_v<typename S::const_reference, const ElementType&>);

  static_assert(S::extent == Size, ""); // check that it exists

  testIterator<S, typename S::iterator>();
  testIterator<S, typename S::reverse_iterator>();
}

template <typename T>
__host__ __device__ void test()
{
  testSpan<cuda::std::span<T>, T, cuda::std::dynamic_extent>();
  testSpan<cuda::std::span<const T>, const T, cuda::std::dynamic_extent>();
  testSpan<cuda::std::span<volatile T>, volatile T, cuda::std::dynamic_extent>();
  testSpan<cuda::std::span<const volatile T>, const volatile T, cuda::std::dynamic_extent>();

  testSpan<cuda::std::span<T, 5>, T, 5>();
  testSpan<cuda::std::span<const T, 5>, const T, 5>();
  testSpan<cuda::std::span<volatile T, 5>, volatile T, 5>();
  testSpan<cuda::std::span<const volatile T, 5>, const volatile T, 5>();
}

struct A
{};

int main(int, char**)
{
  test<int>();
  test<long>();
  test<double>();
  test<A>();

  return 0;
}
