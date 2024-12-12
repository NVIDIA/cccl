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

// cuda::std::views::elements<N>
// cuda::std::views::keys
// cuda::std::views::values

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

template <class T>
struct View : cuda::std::ranges::view_base
{
  __host__ __device__ T* begin() const;
  __host__ __device__ T* end() const;
};

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::elements<0>) )>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::elements<0>) ), View<int>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::elements<0>) ), View<cuda::std::pair<int, int>>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::elements<0>) ), View<cuda::std::tuple<int>>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::elements<5>) ), View<cuda::std::tuple<int>>>);

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::keys))>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::keys)), View<int>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::keys)), View<cuda::std::pair<int, int>>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::keys)), View<cuda::std::tuple<int>>>);

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::values))>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::values)), View<int>>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::values)), View<cuda::std::pair<int, int>>>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::values)), View<cuda::std::tuple<int>>>);

#if TEST_STD_VER >= 2020
template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { cuda::std::forward<View>(view) | cuda::std::forward<T>(t) };
};
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class View, class T, class = void>
inline constexpr bool CanBePiped = false;

template <class View, class T>
inline constexpr bool
  CanBePiped<View, T, cuda::std::void_t<decltype(cuda::std::declval<View>() | cuda::std::declval<T>())>> = true;
#endif // TEST_STD_VER <= 2017
static_assert(!CanBePiped<View<int>, decltype((cuda::std::views::elements<0>) )>);
static_assert(CanBePiped<View<cuda::std::pair<int, int>>, decltype((cuda::std::views::elements<0>) )>);
static_assert(CanBePiped<View<cuda::std::tuple<int>>, decltype((cuda::std::views::elements<0>) )>);
static_assert(!CanBePiped<View<cuda::std::tuple<int>>, decltype((cuda::std::views::elements<5>) )>);

static_assert(!CanBePiped<View<int>, decltype((cuda::std::views::keys))>);
static_assert(CanBePiped<View<cuda::std::pair<int, int>>, decltype((cuda::std::views::keys))>);
static_assert(CanBePiped<View<cuda::std::tuple<int>>, decltype((cuda::std::views::keys))>);

static_assert(!CanBePiped<View<int>, decltype((cuda::std::views::values))>);
static_assert(CanBePiped<View<cuda::std::pair<int, int>>, decltype((cuda::std::views::values))>);
static_assert(!CanBePiped<View<cuda::std::tuple<int>>, decltype((cuda::std::views::values))>);

template <class Range, class Expected>
__host__ __device__ constexpr bool equal(Range&& range, Expected&& expected)
{
  auto irange    = range.begin();
  auto iexpected = cuda::std::begin(expected);
  for (; irange != range.end(); ++irange, ++iexpected)
  {
    if (*irange != *iexpected)
    {
      return false;
    }
  }
  return true;
}

__host__ __device__ constexpr bool test()
{
  cuda::std::pair<int, int> buff[] = {{1, 2}, {3, 4}, {5, 6}};

  // Test `views::elements<N>(v)`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 0>;
    decltype(auto) result = cuda::std::views::elements<0>(buff);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {1, 3, 5};
    assert(equal(result, expected));
  }

  // Test `views::keys(v)`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 0>;
    decltype(auto) result = cuda::std::views::keys(buff);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {1, 3, 5};
    assert(equal(result, expected));
  }

  // Test `views::values(v)`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 1>;
    decltype(auto) result = cuda::std::views::values(buff);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {2, 4, 6};
    assert(equal(result, expected));
  }

  // Test `v | views::elements<N>`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 1>;
    decltype(auto) result = buff | cuda::std::views::elements<1>;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {2, 4, 6};
    assert(equal(result, expected));
  }

  // Test `v | views::keys`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 0>;
    decltype(auto) result = buff | cuda::std::views::keys;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {1, 3, 5};
    assert(equal(result, expected));
  }

  // Test `v | views::values`
  {
    using Result = cuda::std::ranges::elements_view<cuda::std::ranges::ref_view<cuda::std::pair<int, int>[3]>, 1>;
    decltype(auto) result = buff | cuda::std::views::values;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {2, 4, 6};
    assert(equal(result, expected));
  }

  // Test views::elements<0> | views::elements<0>
  {
    cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int>> nested[] = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
    using Result                                                           = cuda::std::ranges::elements_view<
                                                                cuda::std::ranges::
                                                                  elements_view<cuda::std::ranges::ref_view<cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int>>[3]>, 0>,
                                                                0>;
    auto const partial    = cuda::std::views::elements<0> | cuda::std::views::elements<0>;
    decltype(auto) result = nested | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {1, 3, 5};
    assert(equal(result, expected));
  }

  // Test views::keys | views::keys
  {
    cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int>> nested[] = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
    using Result                                                           = cuda::std::ranges::elements_view<
                                                                cuda::std::ranges::
                                                                  elements_view<cuda::std::ranges::ref_view<cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int>>[3]>, 0>,
                                                                0>;
    auto const partial    = cuda::std::views::keys | cuda::std::views::keys;
    decltype(auto) result = nested | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {1, 3, 5};
    assert(equal(result, expected));
  }

  // Test views::values | views::values
  {
    cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int, int>> nested[] = {
      {{1}, {2, 3}}, {{4}, {5, 6}}, {{7}, {8, 9}}};
    using Result = cuda::std::ranges::elements_view<
      cuda::std::ranges::elements_view<
        cuda::std::ranges::ref_view<cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int, int>>[3]>,
        1>,
      1>;
    auto const partial    = cuda::std::views::values | cuda::std::views::values;
    decltype(auto) result = nested | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 6, 9};
    assert(equal(result, expected));
  }

  // Test views::keys | views::values
  {
    cuda::std::pair<cuda::std::tuple<int, int>, cuda::std::tuple<int>> nested[] = {
      {{1, 2}, {3}}, {{4, 5}, {6}}, {{7, 8}, {9}}};
    using Result = cuda::std::ranges::elements_view<
      cuda::std::ranges::elements_view<
        cuda::std::ranges::ref_view<cuda::std::pair<cuda::std::tuple<int, int>, cuda::std::tuple<int>>[3]>,
        0>,
      1>;
    auto const partial    = cuda::std::views::keys | cuda::std::views::values;
    decltype(auto) result = nested | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {2, 5, 8};
    assert(equal(result, expected));
  }

  // Test views::values | views::keys
  {
    cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int, int>> nested[] = {
      {{1}, {2, 3}}, {{4}, {5, 6}}, {{7}, {8, 9}}};
    using Result = cuda::std::ranges::elements_view<
      cuda::std::ranges::elements_view<
        cuda::std::ranges::ref_view<cuda::std::pair<cuda::std::tuple<int>, cuda::std::tuple<int, int>>[3]>,
        1>,
      0>;
    auto const partial    = cuda::std::views::values | cuda::std::views::keys;
    decltype(auto) result = nested | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {2, 5, 8};
    assert(equal(result, expected));
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)
  static_assert(test());
#endif // defined(_LIBCUDACXX_ADDRESSOF) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3)

  return 0;
}
