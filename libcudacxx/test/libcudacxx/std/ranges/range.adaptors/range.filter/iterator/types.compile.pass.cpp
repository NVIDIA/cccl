//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// (clang-14 || gcc-12 || msvc-19.39) in C++20 tries to erroneously instantiate a bunch of
// default constructors that don't exist because it evaluates the class initializers before
// considering the default constructors requirements clause. It's not possible to selectively
// disable them in this file like the others, so we just disable the compiler entirely.

// UNSUPPORTED: (clang-14 || gcc-12 || msvc-19.39) && c++20

// cuda::std::filter_view::<iterator>::difference_type
// cuda::std::filter_view::<iterator>::value_type
// cuda::std::filter_view::<iterator>::iterator_category
// cuda::std::filter_view::<iterator>::iterator_concept

#include "../types.h"

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"

template <typename T>
_CCCL_CONCEPT HasIteratorCategory = _CCCL_REQUIRES_EXPR((T))(requires(T::iterator_category));

template <class Iterator>
using FilterViewFor = cuda::std::ranges::filter_view<minimal_view<Iterator, sentinel_wrapper<Iterator>>, AlwaysTrue>;

template <class Iterator>
using FilterIteratorFor = cuda::std::ranges::iterator_t<FilterViewFor<Iterator>>;

struct ForwardIteratorWithInputCategory
{
  using difference_type   = int;
  using value_type        = int;
  using iterator_category = cuda::std::input_iterator_tag;
  using iterator_concept  = cuda::std::forward_iterator_tag;
  TEST_FUNC ForwardIteratorWithInputCategory();
  TEST_FUNC ForwardIteratorWithInputCategory& operator++();
  TEST_FUNC ForwardIteratorWithInputCategory operator++(int);
  TEST_FUNC int& operator*() const;
  friend TEST_FUNC bool operator==(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
#if TEST_STD_VER <= 2017
  friend TEST_FUNC bool operator!=(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
#endif
};
static_assert(cuda::std::forward_iterator<ForwardIteratorWithInputCategory>);

template <class Iterator>
TEST_FUNC constexpr void test()
{
  using FilterView     = FilterViewFor<Iterator>;
  using FilterIterator = FilterIteratorFor<Iterator>;
  static_assert(
    cuda::std::is_same_v<typename FilterIterator::value_type, cuda::std::ranges::range_value_t<FilterView>>);
  static_assert(
    cuda::std::is_same_v<typename FilterIterator::difference_type, cuda::std::ranges::range_difference_t<FilterView>>);
};

TEST_FUNC constexpr void test()
{
  // Check that value_type is range_value_t and difference_type is range_difference_t
  {
    test<cpp17_input_iterator<int*>>();
    test<cpp20_input_iterator<int*>>();
    test<forward_iterator<int*>>();
    test<bidirectional_iterator<int*>>();
    test<random_access_iterator<int*>>();
    test<contiguous_iterator<int*>>();
    test<int*>();
  }

  // Check iterator_concept for various categories of ranges
  {
    static_assert(cuda::std::is_same_v<FilterIteratorFor<cpp17_input_iterator<int*>>::iterator_concept,
                                       cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<cpp20_input_iterator<int*>>::iterator_concept,
                                       cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<ForwardIteratorWithInputCategory>::iterator_concept,
                                       cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<forward_iterator<int*>>::iterator_concept,
                                       cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<bidirectional_iterator<int*>>::iterator_concept,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<random_access_iterator<int*>>::iterator_concept,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<contiguous_iterator<int*>>::iterator_concept,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(
      cuda::std::is_same_v<FilterIteratorFor<int*>::iterator_concept, cuda::std::bidirectional_iterator_tag>);
  }

  // Check iterator_category for various categories of ranges
  {
    static_assert(!HasIteratorCategory<FilterIteratorFor<cpp17_input_iterator<int*>>>);
    static_assert(!HasIteratorCategory<FilterIteratorFor<cpp20_input_iterator<int*>>>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<ForwardIteratorWithInputCategory>::iterator_category,
                                       cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<forward_iterator<int*>>::iterator_category,
                                       cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<bidirectional_iterator<int*>>::iterator_category,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<random_access_iterator<int*>>::iterator_category,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<FilterIteratorFor<contiguous_iterator<int*>>::iterator_category,
                                       cuda::std::bidirectional_iterator_tag>);
    static_assert(
      cuda::std::is_same_v<FilterIteratorFor<int*>::iterator_category, cuda::std::bidirectional_iterator_tag>);
  }
}

int main(int, char**)
{
  return 0;
}
