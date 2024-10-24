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

// cuda::std::views::filter

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/initializer_list>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

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

struct NonCopyablePredicate
{
  NonCopyablePredicate(NonCopyablePredicate const&) = delete;
  template <class T>
  __host__ __device__ constexpr bool operator()(T x) const
  {
    return x % 2 == 0;
  }
};

struct Range : cuda::std::ranges::view_base
{
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  __host__ __device__ constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr Iterator begin() const
  {
    return Iterator(begin_);
  }
  __host__ __device__ constexpr Sentinel end() const
  {
    return Sentinel(Iterator(end_));
  }

private:
  int* begin_;
  int* end_;
};

struct Pred1
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

struct Pred2
{
  __host__ __device__ constexpr bool operator()(const int i) noexcept
  {
    return i % 3 == 0;
  }
};

template <typename View>
__host__ __device__ constexpr void compareViews(View v, cuda::std::initializer_list<int> list)
{
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2; ++b1, ++b2)
  {
    assert(*b1 == *b2);
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Test `views::filter(pred)(v)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred1>;
    Range const range(buff, buff + 8);
    Pred1 pred;

    {
      decltype(auto) result = cuda::std::views::filter(pred)(range);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial    = cuda::std::views::filter(pred);
      decltype(auto) result = partial(range);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `v | views::filter(pred)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred1>;
    Range const range(buff, buff + 8);
    Pred1 pred;

    {
      decltype(auto) result = range | cuda::std::views::filter(pred);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial    = cuda::std::views::filter(pred);
      decltype(auto) result = range | partial;
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `views::filter(v, pred)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred1>;
    Range const range(buff, buff + 8);
    Pred1 pred;

    decltype(auto) result = cuda::std::views::filter(range, pred);
    static_assert(cuda::std::same_as<decltype(result), Result>);
    compareViews(result, {0, 2, 4, 6});
  }

  // Test that one can call cuda::std::views::filter with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X
    {};
    auto partial = cuda::std::views::filter(X{});
    unused(partial);
  }

// Older gcc cannot handle the piping sequence
#if !defined(TEST_COMPILER_MSVC_2019) && (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 11)
  // Test `adaptor | views::filter(pred)`
  {
    Range const range(buff, buff + 8);

    {
      using Result          = cuda::std::ranges::filter_view<cuda::std::ranges::filter_view<Range, Pred1>, Pred2>;
      decltype(auto) result = range | cuda::std::views::filter(Pred1{}) | cuda::std::views::filter(Pred2{});
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 6});
    }
    {
      using Result          = cuda::std::ranges::filter_view<cuda::std::ranges::filter_view<Range, Pred1>, Pred2>;
      auto const partial    = cuda::std::views::filter(Pred1{}) | cuda::std::views::filter(Pred2{});
      decltype(auto) result = range | partial;
      static_assert(cuda::std::same_as<decltype(result), Result>);
      compareViews(result, {0, 6});
    }
  }
#endif // !msvc && !(gcc < 11)

  // Test SFINAE friendliness
  {
    struct NotAView
    {};
    struct NotInvocable
    {};

    static_assert(!CanBePiped<Range, decltype(cuda::std::views::filter)>);
    static_assert(CanBePiped<Range, decltype(cuda::std::views::filter(Pred1{}))>);
    static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::filter(Pred1{}))>);
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // template instantiation resulted in
                                                                             // unexpected function type
    static_assert(!CanBePiped<Range, decltype(cuda::std::views::filter(NotInvocable{}))>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_ICC

    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter)>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Pred1, Range>);
    static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Range, Pred1>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Range, Pred1, Pred1>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter), NonCopyablePredicate>);
  }

  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::views::filter), decltype(cuda::std::views::filter)>);
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
