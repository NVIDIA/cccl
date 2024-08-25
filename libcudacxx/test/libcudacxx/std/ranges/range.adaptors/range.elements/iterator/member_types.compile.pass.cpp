//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Member typedefs in elements_view<V, N>::iterator.

#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
using Range = cuda::std::ranges::subrange<Iter, sentinel_wrapper<Iter>>;

template <class Range, size_t N = 0>
using ElementsIter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<Range, N>>;

// using iterator_concept = see below;
static_assert(cuda::std::same_as<ElementsIter<Range<cpp20_input_iterator<cuda::std::tuple<int>*>>>::iterator_concept,
                                 cuda::std::input_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<forward_iterator<cuda::std::tuple<int>*>>>::iterator_concept, //
                                 cuda::std::forward_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<bidirectional_iterator<cuda::std::tuple<int>*>>>::iterator_concept,
                                 cuda::std::bidirectional_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<random_access_iterator<cuda::std::tuple<int>*>>>::iterator_concept,
                                 cuda::std::random_access_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<contiguous_iterator<cuda::std::tuple<int>*>>>::iterator_concept,
                                 cuda::std::random_access_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<cuda::std::tuple<int>*>>::iterator_concept, //
                                 cuda::std::random_access_iterator_tag>);

// using iterator_category = see below;   // not always present
#if TEST_STD_VER >= 2020
template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIterCategory = false;

template <class T>
inline constexpr bool HasIterCategory<T, cuda::std::void_t<typename T::iterator_category>> = true;
#endif // TEST_STD_VER <= 2017
static_assert(!HasIterCategory<ElementsIter<Range<cpp20_input_iterator<cuda::std::tuple<int>*>>>>);
static_assert(HasIterCategory<ElementsIter<Range<forward_iterator<cuda::std::tuple<int>*>>>>);

static_assert(cuda::std::same_as<ElementsIter<Range<forward_iterator<cuda::std::tuple<int>*>>>::iterator_category,
                                 cuda::std::forward_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<bidirectional_iterator<cuda::std::tuple<int>*>>>::iterator_category,
                                 cuda::std::bidirectional_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<random_access_iterator<cuda::std::tuple<int>*>>>::iterator_category,
                                 cuda::std::random_access_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<contiguous_iterator<cuda::std::tuple<int>*>>>::iterator_category,
                                 cuda::std::random_access_iterator_tag>);

static_assert(cuda::std::same_as<ElementsIter<Range<cuda::std::tuple<int>*>>::iterator_category, //
                                 cuda::std::random_access_iterator_tag>);

struct ToPair
{
  __host__ __device__ constexpr auto operator()(int) const noexcept
  {
    return cuda::std::pair<int, short>{1, static_cast<short>(1)};
  }
};
using Generator = decltype(cuda::std::views::iota(0, 1) | cuda::std::views::transform(ToPair{}));
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
static_assert(cuda::std::ranges::random_access_range<Generator>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3

static_assert(cuda::std::same_as<ElementsIter<Generator>::iterator_category, //
                                 cuda::std::input_iterator_tag>);

// using value_type = remove_cvref_t<tuple_element_t<N, range_value_t<Base>>>;
static_assert(cuda::std::same_as<ElementsIter<Range<cuda::std::tuple<int, long>*>, 0>::value_type, int>);

static_assert(cuda::std::same_as<ElementsIter<Range<cuda::std::tuple<int, long>*>, 1>::value_type, long>);

static_assert(cuda::std::same_as<ElementsIter<Generator, 0>::value_type, int>);

static_assert(cuda::std::same_as<ElementsIter<Generator, 1>::value_type, short>);

// using difference_type = range_difference_t<Base>;
static_assert(cuda::std::same_as<ElementsIter<Range<cuda::std::tuple<int, long>*>>::difference_type,
                                 cuda::std::ranges::range_difference_t<Range<cuda::std::tuple<int, long>*>>>);

static_assert(cuda::std::same_as<ElementsIter<Generator>::difference_type, //
                                 cuda::std::ranges::range_difference_t<Generator>>);

int main(int, char**)
{
  return 0;
}
