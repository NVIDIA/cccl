//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// cuda::std::filter_view::<iterator>::difference_type
// cuda::std::filter_view::<iterator>::value_type
// cuda::std::filter_view::<iterator>::iterator_category
// cuda::std::filter_view::<iterator>::iterator_concept

#include "../types.h"

#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <typename T>
concept HasIteratorCategory = requires { typename T::iterator_category; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIteratorCategory = false;

template <class T>
inline constexpr bool HasIteratorCategory<T, cuda::std::void_t<typename T::iterator_category>> = true;
#endif // TEST_STD_VER <= 2017

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
  __host__ __device__ ForwardIteratorWithInputCategory();
  __host__ __device__ ForwardIteratorWithInputCategory& operator++();
  __host__ __device__ ForwardIteratorWithInputCategory operator++(int);
  __host__ __device__ int& operator*() const;
  __host__ __device__ friend bool operator==(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
#if TEST_STD_VER <= 2017
  __host__ __device__ friend bool operator!=(ForwardIteratorWithInputCategory, ForwardIteratorWithInputCategory);
#endif // TEST_STD_VER <= 2017
};
static_assert(cuda::std::forward_iterator<ForwardIteratorWithInputCategory>);

template <class Iterator>
__host__ __device__ constexpr void test_iterator(Iterator)
{
  using FilterView      = FilterViewFor<Iterator>;
  using value_type      = typename FilterIteratorFor<Iterator>::value_type;
  using difference_type = typename FilterIteratorFor<Iterator>::difference_type;

  static_assert(cuda::std::is_same_v<value_type, cuda::std::ranges::range_value_t<FilterView>>);
  static_assert(cuda::std::is_same_v<difference_type, cuda::std::ranges::range_difference_t<FilterView>>);
};

__host__ __device__ void f()
{
  // Check that value_type is range_value_t and difference_type is range_difference_t
  {
    test_iterator(cpp17_input_iterator<int*>{nullptr});
    test_iterator(cpp20_input_iterator<int*>{nullptr});
    test_iterator(forward_iterator<int*>{nullptr});
    test_iterator(bidirectional_iterator<int*>{nullptr});
    test_iterator(random_access_iterator<int*>{nullptr});
    test_iterator(contiguous_iterator<int*>{nullptr});
    test_iterator(static_cast<int*>(nullptr));
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
