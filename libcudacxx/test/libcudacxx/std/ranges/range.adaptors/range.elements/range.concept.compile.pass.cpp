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

// concept checking
//
// template<class T, size_t N>
// concept has-tuple-element =
//   tuple-like<T> && N < tuple_size_v<T>;
//
// template<class T, size_t N>
// concept returnable-element =
//   is_reference_v<T> || move_constructible<tuple_element_t<N, T>>;
//
// template<input_range V, size_t N>
//   requires view<V> && has-tuple-element<range_value_t<V>, N> &&
//            has-tuple-element<remove_reference_t<range_reference_t<V>>, N> &&
//            returnable-element<range_reference_t<V>, N>
// class elements_view;

#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_iterators.h"

template <class It>
using Range = cuda::std::ranges::subrange<It, sentinel_wrapper<It>>;

#if TEST_STD_VER >= 2020
template <class V, size_t N>
concept HasElementsView = requires { typename cuda::std::ranges::elements_view<V, N>; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class V, size_t N, class = void>
inline constexpr bool HasElementsView = false;

template <class V, size_t N>
inline constexpr bool HasElementsView<V, N, cuda::std::void_t<decltype(cuda::std::ranges::elements_view<V, N>())>> =
  true;
#endif // TEST_STD_VER <= 2017
static_assert(HasElementsView<Range<cuda::std::ranges::subrange<int*>*>, 0>);
static_assert(HasElementsView<Range<cuda::std::pair<int, int>*>, 1>);
static_assert(HasElementsView<Range<cuda::std::tuple<int, int, int>*>, 2>);
static_assert(HasElementsView<Range<cuda::std::array<int, 4>*>, 3>);

// !view<V>
static_assert(!cuda::std::ranges::view<cuda::std::array<cuda::std::tuple<int>, 1>>);
static_assert(!HasElementsView<cuda::std::array<cuda::std::tuple<int>, 1>, 0>);

// !input_range
static_assert(!cuda::std::ranges::input_range<Range<cpp20_output_iterator<cuda::std::tuple<int>*>>>);
static_assert(!HasElementsView<Range<cpp20_output_iterator<cuda::std::tuple<int>*>>, 0>);

// !tuple-like
LIBCPP_STATIC_ASSERT(!cuda::std::__tuple_like<int>::value);
static_assert(!HasElementsView<Range<int*>, 1>);

// !(N < tuple_size_v<T>)
static_assert(!(2 < cuda::std::tuple_size_v<cuda::std::pair<int, int>>) );
static_assert(!HasElementsView<Range<cuda::std::pair<int, int>*>, 2>);

// ! (is_reference_v<T> || move_constructible<tuple_element_t<N, T>>)
struct NonMovable
{
  __host__ __device__ constexpr NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};
static_assert(!cuda::std::move_constructible<NonMovable>);

struct ToPair
{
  __host__ __device__ constexpr auto operator()(int) const noexcept
  {
    return cuda::std::pair<NonMovable, int>{1, 1};
  }
};

using NonMovableGenerator = decltype(cuda::std::views::iota(0, 1) | cuda::std::views::transform(ToPair{}));

static_assert(!HasElementsView<NonMovableGenerator, 0>);
static_assert(HasElementsView<NonMovableGenerator, 1>);

int main(int, char**)
{
  return 0;
}
