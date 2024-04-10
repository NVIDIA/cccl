//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class It>
struct find_current : private cuda::std::reverse_iterator<It>
{
  __host__ __device__ void test()
  {
    (void) this->current;
  }
};

#if TEST_STD_VER > 2014
template <class It,
          cuda::std::enable_if_t<cuda::std::is_same_v<typename cuda::std::iterator_traits<It>::iterator_category,
                                                      cuda::std::contiguous_iterator_tag>,
                                 int> = 0>
__host__ __device__ constexpr void test_iter_category()
{
  static_assert((cuda::std::is_same<typename cuda::std::move_iterator<It>::iterator_category,
                                    cuda::std::random_access_iterator_tag>::value),
                "");
}

template <class It,
          cuda::std::enable_if_t<!cuda::std::is_same_v<typename cuda::std::iterator_traits<It>::iterator_category,
                                                       cuda::std::contiguous_iterator_tag>,
                                 int> = 0>
__host__ __device__ constexpr void test_iter_category()
{
  static_assert((cuda::std::is_same<typename cuda::std::move_iterator<It>::iterator_category,
                                    typename cuda::std::iterator_traits<It>::iterator_category>::value),
                "");
}
#endif

template <class It>
__host__ __device__ void test()
{
  typedef cuda::std::reverse_iterator<It> R;
  typedef cuda::std::iterator_traits<It> T;
  find_current<It> q;
  q.test(); // Just test that we can access `.current` from derived classes
  static_assert((cuda::std::is_same<typename R::iterator_type, It>::value), "");
  static_assert((cuda::std::is_same<typename R::value_type, typename T::value_type>::value), "");
  static_assert((cuda::std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
  static_assert((cuda::std::is_same<typename R::reference, typename T::reference>::value), "");
  static_assert((cuda::std::is_same<typename R::pointer, typename cuda::std::iterator_traits<It>::pointer>::value), "");

#if TEST_STD_VER <= 2014
  typedef cuda::std::iterator<typename T::iterator_category, typename T::value_type> iterator_base;
  static_assert((cuda::std::is_base_of<iterator_base, R>::value), "");
#endif
#if TEST_STD_VER > 2014
  test_iter_category<It>();
#else
  static_assert((cuda::std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
#endif
}

#if TEST_STD_VER > 2014

struct FooIter
{
  using iterator_category = cuda::std::bidirectional_iterator_tag;
  using value_type        = void*;
  using difference_type   = void*;
  using pointer           = void*;
  using reference         = int&;
  __host__ __device__ int& operator*() const;
  __host__ __device__ FooIter& operator++();
  __host__ __device__ FooIter& operator--();
  __host__ __device__ FooIter operator++(int);
  __host__ __device__ FooIter operator--(int);
};
template <>
struct cuda::std::indirectly_readable_traits<FooIter>
{
  using value_type = int;
};
template <>
struct cuda::std::incrementable_traits<FooIter>
{
  using difference_type = char;
};

// Not using `FooIter::value_type`.
static_assert(cuda::std::is_same_v<typename cuda::std::reverse_iterator<FooIter>::value_type, int>, "");
// Not using `FooIter::difference_type`.
static_assert(cuda::std::is_same_v<typename cuda::std::reverse_iterator<FooIter>::difference_type, char>, "");

#endif

struct BarIter
{
  __host__ __device__ bool& operator*() const;
  __host__ __device__ BarIter& operator++();
  __host__ __device__ BarIter& operator--();
  __host__ __device__ BarIter operator++(int);
  __host__ __device__ BarIter operator--(int);
};
template <>
struct cuda::std::iterator_traits<BarIter>
{
  using difference_type   = char;
  using value_type        = char;
  using pointer           = char*;
  using reference         = char&;
  using iterator_category = cuda::std::bidirectional_iterator_tag;
};

#if TEST_STD_VER > 2014
static_assert(cuda::std::is_same_v<typename cuda::std::reverse_iterator<BarIter>::reference, bool&>, "");
#else
static_assert(cuda::std::is_same<typename cuda::std::reverse_iterator<BarIter>::reference, char&>::value, "");
#endif

__host__ __device__ void test_all()
{
  test<bidirectional_iterator<char*>>();
  test<random_access_iterator<char*>>();
  test<char*>();

#if TEST_STD_VER > 2014
  test<contiguous_iterator<char*>>();
  static_assert(
    cuda::std::is_same_v<typename cuda::std::reverse_iterator<bidirectional_iterator<char*>>::iterator_concept,
                         cuda::std::bidirectional_iterator_tag>,
    "");
  static_assert(
    cuda::std::is_same_v<typename cuda::std::reverse_iterator<random_access_iterator<char*>>::iterator_concept,
                         cuda::std::random_access_iterator_tag>,
    "");
  static_assert(
    cuda::std::is_same_v<typename cuda::std::reverse_iterator<cpp20_random_access_iterator<char*>>::iterator_concept,
                         cuda::std::random_access_iterator_tag>,
    "");
  static_assert(cuda::std::is_same_v<typename cuda::std::reverse_iterator<contiguous_iterator<char*>>::iterator_concept,
                                     cuda::std::random_access_iterator_tag>,
                "");
  static_assert(cuda::std::is_same_v<typename cuda::std::reverse_iterator<char*>::iterator_concept,
                                     cuda::std::random_access_iterator_tag>,
                "");
#endif
}

int main(int, char**)
{
  return 0;
}
