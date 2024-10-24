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

// cuda::std::views::common

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"
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

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buf[] = {1, 2, 3};

  // views::common(r) is equivalent to views::all(r) if r is a common_range
  {
    {
      CommonView view(buf, buf + 3);
      decltype(auto) result = cuda::std::views::common(view);
      static_assert(cuda::std::same_as<decltype(result), CommonView>);
      assert(result.begin_ == buf);
      assert(result.end_ == buf + 3);
    }
    {
      using NotAView        = cuda::std::array<int, 3>;
      NotAView arr          = {1, 2, 3};
      decltype(auto) result = cuda::std::views::common(arr);
      static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::ref_view<NotAView>>);
      assert(result.begin() == arr.begin());
      assert(result.end() == arr.end());
    }
  }

  // Otherwise, views::common(r) is equivalent to ranges::common_view{r}
  {
    NonCommonView view(buf, buf + 3);
    decltype(auto) result = cuda::std::views::common(view);
    static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::common_view<NonCommonView>>);
    assert(result.base().begin_ == buf);
    assert(result.base().end_ == buf + 3);
  }

  // Test that cuda::std::views::common is a range adaptor
  {
    using SomeView = NonCommonView;

    // Test `v | views::common`
    {
      SomeView view(buf, buf + 3);
      decltype(auto) result = view | cuda::std::views::common;
      static_assert(cuda::std::same_as<decltype(result), cuda::std::ranges::common_view<SomeView>>);
      assert(result.base().begin_ == buf);
      assert(result.base().end_ == buf + 3);
    }

    // Test `adaptor | views::common`
    {
      SomeView view(buf, buf + 3);
      auto f = [](int i) {
        return i;
      };
      auto const partial    = cuda::std::views::transform(f) | cuda::std::views::common;
      using Result          = cuda::std::ranges::common_view<cuda::std::ranges::transform_view<SomeView, decltype(f)>>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + 3);
    }

    // Test `views::common | adaptor`
    {
      SomeView view(buf, buf + 3);
      auto f = [](int i) {
        return i;
      };
      auto const partial    = cuda::std::views::common | cuda::std::views::transform(f);
      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::common_view<SomeView>, decltype(f)>;
      decltype(auto) result = partial(view);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(result.base().base().begin_ == buf);
      assert(result.base().base().end_ == buf + 3);
    }

    // Check SFINAE friendliness
    {
      struct NotAView
      {};
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::common)>);
      static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::common), NotAView>);
      static_assert(CanBePiped<SomeView&, decltype(cuda::std::views::common)>);
      static_assert(CanBePiped<int(&)[10], decltype(cuda::std::views::common)>);
      static_assert(!CanBePiped<int(&&)[10], decltype(cuda::std::views::common)>);
      static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::common)>);
    }
  }

  {
    static_assert(cuda::std::same_as<decltype(cuda::std::views::common), decltype(cuda::std::ranges::views::common)>);
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
