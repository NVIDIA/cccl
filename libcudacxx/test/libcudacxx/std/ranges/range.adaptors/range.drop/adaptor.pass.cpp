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

// cuda::std::views::drop

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/span>
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  include <cuda/std/string_view>
#endif
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

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

struct SizedViewWithUnsizedSentinel : cuda::std::ranges::view_base
{
  using iterator = random_access_iterator<int*>;
  using sentinel = sentinel_wrapper<random_access_iterator<int*>>;

  int* begin_ = nullptr;
  int* end_   = nullptr;
  __host__ __device__ constexpr SizedViewWithUnsizedSentinel(int* begin, int* end)
      : begin_(begin)
      , end_(end)
  {}

  __host__ __device__ constexpr auto begin() const
  {
    return iterator(begin_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sentinel(iterator(end_));
  }
  __host__ __device__ constexpr size_t size() const
  {
    return end_ - begin_;
  }
};
static_assert(cuda::std::ranges::random_access_range<SizedViewWithUnsizedSentinel>);
static_assert(cuda::std::ranges::sized_range<SizedViewWithUnsizedSentinel>);
static_assert(
  !cuda::std::sized_sentinel_for<SizedViewWithUnsizedSentinel::sentinel, SizedViewWithUnsizedSentinel::iterator>);
static_assert(cuda::std::ranges::view<SizedViewWithUnsizedSentinel>);

template <class T>
__host__ __device__ constexpr void test_small_range(const T& input)
{
  constexpr int N = 100;
  auto size       = cuda::std::ranges::size(input);
  assert(size < N);

  auto result = input | cuda::std::views::drop(N);
  assert(result.empty());
}

struct Pred
{
  __host__ __device__ constexpr int operator()(int i) noexcept
  {
    return i;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  constexpr int N = 8;
  int buf[N]      = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test that `cuda::std::views::drop` is a range adaptor.
  {
    using SomeView = SizedView;

    // Test `view | views::drop`
    {
      SomeView view(buf, buf + N);
      decltype(auto) result = view | cuda::std::views::drop(3);
      static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::drop_view<SomeView>>);
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + N);
      assert(base(result.begin()) == buf + 3);
      assert(base(base(result.end())) == buf + N);
      assert(result.size() == 5);
    }

    // Test `adaptor | views::drop`
    {
      SomeView view(buf, buf + N);
      auto const partial = cuda::std::views::transform(Pred{}) | cuda::std::views::drop(3);

      using Result          = cuda::std::ranges::drop_view<cuda::std::ranges::transform_view<SomeView, Pred>>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(base(result.begin().base()) == buf + 3);
      assert(base(base(result.end().base())) == buf + N);
      assert(result.size() == 5);
    }

    // Test `views::drop | adaptor`
    {
      SomeView view(buf, buf + N);
      auto const partial = cuda::std::views::drop(3) | cuda::std::views::transform(Pred{});

      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::drop_view<SomeView>, Pred>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + N);
      assert(base(result.begin().base()) == buf + 3);
      assert(base(base(result.end().base())) == buf + N);
      assert(result.size() == 5);
    }

    // Check SFINAE friendliness
    {
      struct NotAView
      {};
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::drop)>);
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::drop), NotAView, int>);
      static_assert(CanBePiped<SomeView&, decltype(cuda::std::views::drop(3))>);
      static_assert(CanBePiped<int(&)[10], decltype(cuda::std::views::drop(3))>);
      static_assert(!CanBePiped<int(&&)[10], decltype(cuda::std::views::drop(3))>);
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // template instantiation resulted in
                                                                             // unexpected function type
      static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::drop(3))>);
      static_assert(!CanBePiped<SomeView&, decltype(cuda::std::views::drop(/*n=*/NotAView{}))>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_ICC
    }
  }

  {
    static_assert(cuda::std::same_as<decltype(cuda::std::views::drop), decltype(cuda::std::ranges::views::drop)>);
  }

  // `views::drop(empty_view, n)` returns an `empty_view`.
  {
    using Result          = cuda::std::ranges::empty_view<int>;
    decltype(auto) result = cuda::std::views::empty<int> | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    unused(result);
  }

  // `views::drop(span, n)` returns a `span`.
  {
    cuda::std::span<int> s(buf);
    decltype(auto) result = s | cuda::std::views::drop(5);
    static_assert(cuda::std::same_as<decltype(result), decltype(s)>);
    assert(result.size() == 3);
  }

  // `views::drop(span, n)` returns a `span` with a dynamic extent, regardless of the input `span`.
  {
    cuda::std::span<int, 8> s(buf);
    decltype(auto) result = s | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), cuda::std::span<int, cuda::std::dynamic_extent>>);
    assert(result.size() == 5);
  }

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
  // `views::drop(string_view, n)` returns a `string_view`.
  {
    {
      cuda::std::string_view sv = "abcdef";
      decltype(auto) result     = sv | cuda::std::views::drop(2);
      static_assert(cuda::std::same_as<decltype(result), decltype(sv)>);
      assert(result.size() == 4);
    }

    {
      cuda::std::u32string_view sv = U"abcdef";
      decltype(auto) result        = sv | cuda::std::views::drop(2);
      static_assert(cuda::std::same_as<decltype(result), decltype(sv)>);
      assert(result.size() == 4);
    }
  }
#endif

  // `views::drop(iota_view, n)` returns an `iota_view`.
  {
    auto iota = cuda::std::views::iota(1, 8);
    // The second template argument of the resulting `iota_view` is different because it has to be able to hold
    // the `range_difference_t` of the input `iota_view`.
    using Result          = cuda::std::ranges::iota_view<int, int>;
    decltype(auto) result = iota | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    assert(result.size() == 4);
    assert(*result.begin() == 4);
    assert(*cuda::std::ranges::next(result.begin(), 3) == 7);
  }

  // `views::drop(subrange, n)` returns a `subrange` when `subrange::StoreSize == false`.
  {
    cuda::std::ranges::subrange<int*> subrange(buf, buf + N);
    LIBCPP_STATIC_ASSERT(!decltype(subrange)::_StoreSize);

    decltype(auto) result = subrange | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), decltype(subrange)>);
    assert(result.size() == 5);
  }

  // `views::drop(subrange, n)` returns a `subrange` when `subrange::StoreSize == true`.
  {
    using View = SizedViewWithUnsizedSentinel;
    View view(buf, buf + N);

    using Subrange =
      cuda::std::ranges::subrange<View::iterator, View::sentinel, cuda::std::ranges::subrange_kind::sized>;
    auto subrange = Subrange(view.begin(), view.end(), cuda::std::ranges::distance(view.begin(), view.end()));
    LIBCPP_STATIC_ASSERT(decltype(subrange)::_StoreSize);

    decltype(auto) result = subrange | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), Subrange>);
    assert(result.size() == 5);
  }

  // `views::drop(subrange, n)` doesn't return a `subrange` if it's not a random access range.
  {
    SizedView v(buf, buf + N);
    auto subrange = cuda::std::ranges::subrange(v.begin(), v.end());

    using Result = cuda::std::ranges::drop_view<
      cuda::std::ranges::subrange<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>>;
    decltype(auto) result = subrange | cuda::std::views::drop(3);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    assert(result.size() == 5);
  }

  // When the size of the input range `s` is shorter than `n`, an `empty_view` is returned.
  {
    test_small_range(cuda::std::span(buf));
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
    test_small_range(cuda::std::string_view("abcdef"));
#endif
    test_small_range(cuda::std::ranges::subrange(buf, buf + N));
    test_small_range(cuda::std::views::iota(1, 8));
  }

  // Test that it's possible to call `cuda::std::views::drop` with any single argument as long as the resulting closure
  // is never invoked. There is no good use case for it, but it's valid.
  {
    struct X
    {};
    auto partial = cuda::std::views::drop(X{});
    unused(partial);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
