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

// cuda::std::views::take

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/span>
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  include <cuda/std/string_view>
#endif
#include <cuda/std/utility>

#include "test_iterators.h"

#if TEST_STD_VER > 2017
template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { cuda::std::forward<View>(view) | cuda::std::forward<T>(t) };
};
#else
template <class View, class T, class = void>
inline constexpr bool CanBePiped = false;

template <class View, class T>
inline constexpr bool
  CanBePiped<View, T, cuda::std::void_t<decltype(cuda::std::declval<View>() | cuda::std::declval<T>())>> = true;
#endif

struct SizedView : cuda::std::ranges::view_base
{
  int* begin_ = nullptr;
  int* end_   = nullptr;
  __host__ __device__ constexpr SizedView(int* begin, int* end)
      : begin_(begin)
      , end_(end)
  {}

  __host__ __device__ constexpr auto begin() const
  {
    return forward_iterator<int*>(begin_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<forward_iterator<int*>>(forward_iterator<int*>(end_));
  }
};
static_assert(cuda::std::ranges::forward_range<SizedView>);
static_assert(cuda::std::ranges::sized_range<SizedView>);
static_assert(cuda::std::ranges::view<SizedView>);

template <class T>
__host__ __device__ constexpr void test_small_range(const T& input)
{
  constexpr int N = 100;
  auto size       = cuda::std::ranges::size(input);

  auto result = input | cuda::std::views::take(N);
  assert(size < N);
  assert(result.size() == size);
}

struct Pred
{
  __host__ __device__ int operator()(int i) const noexcept
  {
    return i;
  }
};

// GCC really hates aliases defined inside of functions
using result_subrange       = cuda::std::ranges::subrange<int*>;
using result_subrange_sized = cuda::std::ranges::subrange<int*, int*, cuda::std::ranges::subrange_kind::sized>;

__host__ __device__ constexpr bool test()
{
  constexpr int N = 8;
  int buf[N]      = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test that `cuda::std::views::take` is a range adaptor.
  {
    using SomeView = SizedView;

    // Test `view | views::take`
    {
      SomeView view(buf, buf + N);
      decltype(auto) result = view | cuda::std::views::take(3);
      static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::take_view<SomeView>>);
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + N);
      assert(result.size() == 3);
    }

    // Test `adaptor | views::take`
    {
      SomeView view(buf, buf + N);
      auto const partial = cuda::std::views::transform(Pred{}) | cuda::std::views::take(3);

      using Result          = cuda::std::ranges::take_view<cuda::std::ranges::transform_view<SomeView, Pred>>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(result.size() == 3);
    }

    // Test `views::take | adaptor`
    {
      SomeView view(buf, buf + N);
      auto const partial = cuda::std::views::take(3) | cuda::std::views::transform(Pred{});

      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::take_view<SomeView>, Pred>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(result.size() == 3);
    }

    // Check SFINAE friendliness
    {
      struct NotAView
      {};
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::take)>);
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::take), NotAView, int>);
      static_assert(CanBePiped<SomeView&, decltype(cuda::std::views::take(3))>);
      static_assert(CanBePiped<int(&)[10], decltype(cuda::std::views::take(3))>);
      static_assert(!CanBePiped<int(&&)[10], decltype(cuda::std::views::take(3))>);
      static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::take(3))>);

#if !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_NVRTC) // ICE
      static_assert(!CanBePiped<SomeView&, decltype(cuda::std::views::take(/*n=*/NotAView{}))>);
#endif // !TEST_COMPILER_NVCC && !TEST_COMPILER_NVRTC
    }
  }

  {
    static_assert(cuda::std::same_as<decltype(cuda::std::views::take), decltype(cuda::std::ranges::views::take)>);
  }

  // `views::take(empty_view, n)` returns an `empty_view`.
  {
    using Result          = cuda::std::ranges::empty_view<int>;
    decltype(auto) result = cuda::std::views::empty<int> | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    unused(result);
  }

  // `views::take(span, n)` returns a `span`.
  {
    cuda::std::span<int> s(buf);
    decltype(auto) result = s | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), decltype(s)>);
    assert(result.size() == 3);
  }

  // `views::take(span, n)` returns a `span` with a dynamic extent, regardless of the input `span`.
  {
    cuda::std::span<int, 8> s(buf);
    decltype(auto) result = s | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), cuda::std::span<int, cuda::std::dynamic_extent>>);
    assert(result.size() == 3);
  }

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
  // `views::take(string_view, n)` returns a `string_view`.
  {
    {
      cuda::std::string_view sv = "abcdef";
      decltype(auto) result     = sv | cuda::std::views::take(3);
      static_assert(cuda::std::same_as<decltype(result), decltype(sv)>);
      assert(result.size() == 3);
    }

    {
      cuda::std::u32string_view sv = U"abcdef";
      decltype(auto) result        = sv | cuda::std::views::take(3);
      static_assert(cuda::std::same_as<decltype(result), decltype(sv)>);
      assert(result.size() == 3);
    }
  }
#endif // _LIBCUDACXX_HAS_STRING_VIEW

  // `views::take(subrange, n)` returns a `subrange`.
  {
    auto subrange         = cuda::std::ranges::subrange(buf, buf + N);
    decltype(auto) result = subrange | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), result_subrange>);
    assert(result.size() == 3);
  }

  // `views::take(subrange, n)` doesn't return a `subrange` if it's not a random access range.
  {
    SizedView v(buf, buf + N);
    auto subrange = cuda::std::ranges::subrange(v.begin(), v.end());

    using Result = cuda::std::ranges::take_view<
      cuda::std::ranges::subrange<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>>;
    decltype(auto) result = subrange | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    assert(result.size() == 3);
  }

  // `views::take(subrange, n)` returns a `subrange` with all default template arguments.
  {
    cuda::std::ranges::subrange<int*, sized_sentinel<int*>, cuda::std::ranges::subrange_kind::sized> subrange;

    decltype(auto) result = subrange | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), result_subrange_sized>);
    unused(result);
  }

  // `views::take(iota_view, n)` returns an `iota_view`.
  {
    auto iota = cuda::std::views::iota(1, 8);
    // The second template argument of the resulting `iota_view` is different because it has to be able to hold
    // the `range_difference_t` of the input `iota_view`.
    using Result          = cuda::std::ranges::iota_view<int, cuda::std::ranges::range_difference_t<decltype(iota)>>;
    decltype(auto) result = iota | cuda::std::views::take(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    assert(result.size() == 3);
  }
  // When the size of the input range `s` is shorter than `n`, only `s` elements are taken.
  {
    test_small_range(cuda::std::span(buf));
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
    test_small_range(cuda::std::string_view("abcdef"));
#endif
    test_small_range(cuda::std::ranges::subrange(buf, buf + N));
    test_small_range(cuda::std::views::iota(1, 8));
  }

  // Test that it's possible to call `cuda::std::views::take` with any single argument as long as the resulting closure
  // is never invoked. There is no good use case for it, but it's valid.
  {
    struct X
    {};
    auto partial = cuda::std::views::take(X{});
    unused(partial);
  }

// Test when `subrange<Iter>` is not well formed
#if TEST_STD_VER > 2017 || !defined(__clang__) // clang crashes here checking the constraints
  {
    int input[] = {1, 2, 3};
    using Iter  = cpp20_input_iterator<int*>;
    using Sent  = sentinel_wrapper<Iter>;
    cuda::std::ranges::subrange r{Iter{input}, Sent{Iter{input + 3}}};
    auto tv = cuda::std::views::take(cuda::std::move(r), 1);
    auto it = tv.begin();
    assert(*it == 1);
    ++it;
    assert(it == tv.end());
  }
#endif

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test(), "");
#endif

  return 0;
}
