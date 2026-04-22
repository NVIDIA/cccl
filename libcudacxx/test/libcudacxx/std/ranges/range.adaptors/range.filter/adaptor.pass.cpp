//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// cuda::std::views::filter

#include <cuda/std/concepts>
#include <cuda/std/initializer_list>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class View, class T>
_CCCL_CONCEPT CanBePiped =
  _CCCL_REQUIRES_EXPR((View, T), View&& view, T&& t)((cuda::std::forward<View>(view) | cuda::std::forward<T>(t)));

struct NonCopyablePredicate
{
  NonCopyablePredicate(NonCopyablePredicate const&)            = delete;
  NonCopyablePredicate& operator=(NonCopyablePredicate const&) = delete;

  template <class T>
  [[nodiscard]] TEST_FUNC constexpr bool operator()(const T& x) const
  {
    return x % 2 == 0;
  }
};

struct Range : cuda::std::ranges::view_base
{
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;

  // (clang-14 || gcc-12 || msvc-19.39) in C++20 tries to erroneously instantiate a bunch of
  // default constructors that don't exist because it evaluates the class initializers before
  // considering the default constructors requirements clause.
#if (TEST_COMPILER(CLANG, ==, 14) || TEST_COMPILER(GCC, ==, 12) || TEST_COMPILER(MSVC, <, 19, 44)) \
  && (TEST_STD_VER == 2020)
  TEST_FUNC constexpr explicit Range()
      : Range{nullptr, nullptr}
  {}
#endif // (TEST_COMPILER(CLANG, ==, 14) || TEST_COMPILER(GCC, ==, 12)
       // || TEST_COMPILER(MSVC, <, 19, 44)) && (TEST_STD_VER == 2020)

  TEST_FUNC constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  [[nodiscard]] TEST_FUNC constexpr Iterator begin() const
  {
    return Iterator(begin_);
  }
  [[nodiscard]] TEST_FUNC constexpr Sentinel end() const
  {
    return Sentinel(Iterator(end_));
  }

private:
  int* begin_;
  int* end_;
};

struct Pred
{
  [[nodiscard]] TEST_FUNC constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

template <typename View>
TEST_FUNC constexpr void compareViews(View v, cuda::std::initializer_list<int> list)
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

TEST_FUNC constexpr bool test()
{
  int buff[] = {0, 1, 2, 3, 4, 5, 6, 7};

  // Test `views::filter(pred)(v)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred{};

    {
      auto result = cuda::std::views::filter(pred)(range);
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial = cuda::std::views::filter(pred);
      auto result        = partial(range);
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `v | views::filter(pred)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred{};

    {
      auto result = range | cuda::std::views::filter(pred);
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 2, 4, 6});
    }
    {
      auto const partial = cuda::std::views::filter(pred);
      auto result        = range | partial;
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 2, 4, 6});
    }
  }

  // Test `views::filter(v, pred)`
  {
    using Result = cuda::std::ranges::filter_view<Range, Pred>;
    Range const range(buff, buff + 8);
    Pred pred{};

    auto result = cuda::std::views::filter(range, pred);
    static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
    compareViews(result, {0, 2, 4, 6});
  }

  // Test that one can call cuda::std::views::filter with arbitrary stuff, as long as we
  // don't try to actually complete the call by passing it a range.
  //
  // That makes no sense and we can't do anything with the result, but it's valid.
  {
    struct X
    {};
    [[maybe_unused]] auto partial = cuda::std::views::filter(X{});
  }

  // nvcc 12.0 commits harakiri by way of
  //
  // libcudacxx/include/cuda/std/__ranges/filter_view.h(85): error: Internal Compiler Error
  // (codegen): "internal error during structure layout!"
  //
  // You can fix the ICE by making the predicates constexpr, but then the static_assert()s fail
  // (but also only for nvcc 12.0).
#if !TEST_CUDA_COMPILER(NVCC) || TEST_CUDA_COMPILER(NVCC, >=, 12, 1)
  {
    // Test `adaptor | views::filter(pred)`
    Range const range(buff, buff + 8);

    {
      auto pred1 = [](int i) {
        return i % 2 == 0;
      };
      auto pred2 = [](int i) {
        return i % 3 == 0;
      };
      using Result =
        cuda::std::ranges::filter_view<cuda::std::ranges::filter_view<Range, decltype(pred1)>, decltype(pred2)>;
      auto result = range | cuda::std::views::filter(pred1) | cuda::std::views::filter(pred2);
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 6});
    }
    {
      auto pred1 = [](int i) {
        return i % 2 == 0;
      };
      auto pred2 = [](int i) {
        return i % 3 == 0;
      };
      using Result =
        cuda::std::ranges::filter_view<cuda::std::ranges::filter_view<Range, decltype(pred1)>, decltype(pred2)>;
      auto const partial = cuda::std::views::filter(pred1) | cuda::std::views::filter(pred2);
      auto result        = range | partial;
      static_assert(cuda::std::is_same_v<Result, cuda::std::remove_cvref_t<decltype(result)>>);
      compareViews(result, {0, 6});
    }
  }
#endif // !TEST_CUDA_COMPILER(NVCC) || TEST_CUDA_COMPILER(NVCC, >=, 12, 1)

  // Test SFINAE friendliness
  {
    struct NotAView
    {};
    struct NotInvocable
    {};

    static_assert(!CanBePiped<Range, decltype(cuda::std::views::filter)>);
    static_assert(CanBePiped<Range, decltype(cuda::std::views::filter(Pred{}))>);
    static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::filter(Pred{}))>);
    static_assert(!CanBePiped<Range, decltype(cuda::std::views::filter(NotInvocable{}))>);

    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter)>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Pred, Range>);
    static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Range, Pred>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::filter), Range, Pred, Pred>);
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
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
