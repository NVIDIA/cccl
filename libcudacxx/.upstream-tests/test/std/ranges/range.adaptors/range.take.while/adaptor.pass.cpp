//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// cuda::std::views::take_while

//#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "types.h"
#include "test_macros.h"

struct Pred {
  __host__ __device__ constexpr bool operator()(int i) const { return i < 3; }
};

struct Foo {};

struct MoveOnlyView : IntBufferViewBase {
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  MoveOnlyView() = default;

  template <class T>
  __host__ __device__ constexpr
  MoveOnlyView(T&& input) : IntBufferViewBase(cuda::std::forward<T>(input)) {}
#else // ^^^ TEST_COMPILER_NVRTC ^^^ / vvv !TEST_COMPILER_NVRTC vvv
  using IntBufferViewBase::IntBufferViewBase;
#endif // !TEST_COMPILER_NVRTC
  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&)                 = default;
  MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  __host__ __device__ constexpr const int* begin() const { return buffer_; }
  __host__ __device__ constexpr const int* end() const { return buffer_ + size_; }
};

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::take_while))>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), int>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), Pred>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), int, Pred>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), int (&)[2], Pred>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), Foo (&)[2], Pred>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::take_while)), MoveOnlyView, Pred>);

#if TEST_STD_VER > 17
template <class View, class T>
concept CanBePiped =
    requires(View&& view, T&& t) {
      { cuda::std::forward<View>(view) | cuda::std::forward<T>(t) };
    };
#else
template <class View, class T, class = void>
inline constexpr bool CanBePiped = false;

template <class View, class T>
inline constexpr bool CanBePiped<View, T,
  cuda::std::void_t<decltype(cuda::std::declval<View>() | cuda::std::declval<T>())>> = true;
#endif

static_assert(!CanBePiped<MoveOnlyView, decltype(cuda::std::views::take_while)>);
static_assert(CanBePiped<MoveOnlyView, decltype(cuda::std::views::take_while(Pred{}))>);
static_assert(!CanBePiped<int, decltype(cuda::std::views::take_while(Pred{}))>);
static_assert(CanBePiped<int (&)[2], decltype(cuda::std::views::take_while(Pred{}))>);
#ifndef TEST_COMPILER_NVCC_BELOW_11_3
static_assert(!CanBePiped<Foo (&)[2], decltype(cuda::std::views::take_while(Pred{}))>);
#endif // !TEST_COMPILER_NVCC_BELOW_11_3

template <class Range, class Expected>
__host__ __device__ constexpr bool equal(Range&& range, Expected&& expected) {
  auto irange = range.begin();
  auto iexpected = cuda::std::begin(expected);
  for (; irange != range.end(); ++irange, ++iexpected) {
    if (*irange != *iexpected) {
      return false;
    }
  }
  return true;
}

__host__ __device__ constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 3, 2, 1};

  // Test `views::take_while(p)(v)`
  {
    using Result                               = cuda::std::ranges::take_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = cuda::std::views::take_while(Pred{})(MoveOnlyView{buff});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1, 2};
    assert(equal(result, expected));
  }
  {
    auto const partial                         = cuda::std::views::take_while(Pred{});
    using Result                               = cuda::std::ranges::take_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = partial(MoveOnlyView{buff});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1, 2};
    assert(equal(result, expected));
  }

  // Test `v | views::take_while(p)`
  {
    using Result                               = cuda::std::ranges::take_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = MoveOnlyView{buff} | cuda::std::views::take_while(Pred{});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1, 2};
    assert(equal(result, expected));
  }
  {
    auto const partial                         = cuda::std::views::take_while(Pred{});
    using Result                               = cuda::std::ranges::take_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = MoveOnlyView{buff} | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1, 2};
    assert(equal(result, expected));
  }

  // Test `views::take_while(v, p)`
  {
    using Result                               = cuda::std::ranges::take_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = cuda::std::views::take_while(MoveOnlyView{buff}, Pred{});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1, 2};
    assert(equal(result, expected));
  }

  // Test adaptor | adaptor
  {
    struct Pred2 {
      __host__ __device__ constexpr bool operator()(int i) const { return i < 2; }
    };
    auto const partial = cuda::std::views::take_while(Pred{}) | cuda::std::views::take_while(Pred2{});
    using Result       = cuda::std::ranges::take_while_view<cuda::std::ranges::take_while_view<MoveOnlyView, Pred>, Pred2>;
    decltype(auto) result = MoveOnlyView{buff} | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected                              = {1};
    assert(equal(result, expected));
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 17 && _LIBCUDACXX_ADDRESSOF

  return 0;
}
