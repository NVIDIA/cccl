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

// cuda::std::views::reverse

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "types.h"

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

struct Pred
{
  __host__ __device__ int operator()(int i) const noexcept
  {
    return i;
  }
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buf[] = {1, 2, 3};

  // views::reverse(x) is equivalent to x.base() if x is a reverse_view
  {
    {
      BidirRange view(buf, buf + 3);
      cuda::std::ranges::reverse_view<BidirRange> reversed(view);
      decltype(auto) result = cuda::std::views::reverse(reversed);
      static_assert(cuda::std::same_as<decltype(result), BidirRange>);
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
    {
      // Common use case is worth testing
      BidirRange view(buf, buf + 3);
      decltype(auto) result = cuda::std::views::reverse(cuda::std::views::reverse(view));
      static_assert(cuda::std::same_as<decltype(result), BidirRange>);
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
  }

  // views::reverse(x) is equivalent to subrange{end, begin, size} if x is a
  // sized subrange over reverse iterators
  {
    using It       = bidirectional_iterator<int*>;
    using Subrange = cuda::std::ranges::subrange<It, It, cuda::std::ranges::subrange_kind::sized>;

    using ReverseIt       = cuda::std::reverse_iterator<It>;
    using ReverseSubrange = cuda::std::ranges::subrange<ReverseIt, ReverseIt, cuda::std::ranges::subrange_kind::sized>;

    {
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(
        ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)), /* size */ 3);
      decltype(auto) result = cuda::std::views::reverse(subrange);
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // cuda::std::move into views::reverse
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(
        ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)), /* size */ 3);
      decltype(auto) result = cuda::std::views::reverse(cuda::std::move(subrange));
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // with a const subrange
      BidirRange view(buf, buf + 3);
      ReverseSubrange const subrange(
        ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)), /* size */ 3);
      decltype(auto) result = cuda::std::views::reverse(subrange);
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
  }

  // views::reverse(x) is equivalent to subrange{end, begin} if x is an
  // unsized subrange over reverse iterators
  {
    using It       = bidirectional_iterator<int*>;
    using Subrange = cuda::std::ranges::subrange<It, It, cuda::std::ranges::subrange_kind::unsized>;

    using ReverseIt = cuda::std::reverse_iterator<It>;
    using ReverseSubrange =
      cuda::std::ranges::subrange<ReverseIt, ReverseIt, cuda::std::ranges::subrange_kind::unsized>;

    {
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)));
      decltype(auto) result = cuda::std::views::reverse(subrange);
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // cuda::std::move into views::reverse
      BidirRange view(buf, buf + 3);
      ReverseSubrange subrange(ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)));
      decltype(auto) result = cuda::std::views::reverse(cuda::std::move(subrange));
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
    {
      // with a const subrange
      BidirRange view(buf, buf + 3);
      ReverseSubrange const subrange(ReverseIt(cuda::std::ranges::end(view)), ReverseIt(cuda::std::ranges::begin(view)));
      decltype(auto) result = cuda::std::views::reverse(subrange);
      static_assert(cuda::std::same_as<decltype(result), Subrange>);
      assert(base(result.begin()) == buf);
      assert(base(result.end()) == buf + 3);
    }
  }

  // Otherwise, views::reverse(x) is equivalent to ranges::reverse_view{x}
  {
    BidirRange view(buf, buf + 3);
    decltype(auto) result = cuda::std::views::reverse(view);
    static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::reverse_view<BidirRange>>);
    assert(base(result.begin().base()) == buf + 3);
    assert(base(result.end().base()) == buf);
  }

  // Test that cuda::std::views::reverse is a range adaptor
  {
    // Test `v | views::reverse`
    {
      BidirRange view(buf, buf + 3);
      decltype(auto) result = view | cuda::std::views::reverse;
      static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::reverse_view<BidirRange>>);
      assert(base(result.begin().base()) == buf + 3);
      assert(base(result.end().base()) == buf);
    }

    // Test `adaptor | views::reverse`
    {
      BidirRange view(buf, buf + 3);
      auto const partial    = cuda::std::views::transform(Pred{}) | cuda::std::views::reverse;
      using Result          = cuda::std::ranges::reverse_view<cuda::std::ranges::transform_view<BidirRange, Pred>>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(base(result.begin().base().base()) == buf + 3);
      assert(base(result.end().base().base()) == buf);
    }

    // Test `views::reverse | adaptor`
    {
      BidirRange view(buf, buf + 3);
      auto const partial    = cuda::std::views::reverse | cuda::std::views::transform(Pred{});
      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::reverse_view<BidirRange>, Pred>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(base(result.begin().base().base()) == buf + 3);
      assert(base(result.end().base().base()) == buf);
    }

    // Check SFINAE friendliness
    {
      struct NotABidirRange
      {};
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::reverse)>);
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::reverse), NotABidirRange>);
      static_assert(CanBePiped<BidirRange, decltype(cuda::std::views::reverse)>);
      static_assert(CanBePiped<BidirRange&, decltype(cuda::std::views::reverse)>);
      static_assert(!CanBePiped<NotABidirRange, decltype(cuda::std::views::reverse)>);
    }
  }

  {
    static_assert(cuda::std::same_as<decltype(cuda::std::views::reverse), decltype(cuda::std::ranges::views::reverse)>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
