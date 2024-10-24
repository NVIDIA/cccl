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

// cuda::std::views::drop_while

// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
};

struct Foo
{};

template <class T>
struct BufferView : cuda::std::ranges::view_base
{
  T* buffer_;
  cuda::std::size_t size_;

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferView(T (&b)[N])
      : buffer_(b)
      , size_(N)
  {}
};

using IntBufferView = BufferView<int>;

struct MoveOnlyView : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC)
  MoveOnlyView() noexcept = default;

  template <class T>
  __host__ __device__ constexpr MoveOnlyView(T&& arr) noexcept(noexcept(IntBufferView(cuda::std::declval<T>())))
      : IntBufferView(_CUDA_VSTD::forward<T>(arr))
  {}
#else
  using IntBufferView::IntBufferView;
#endif

  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&)                 = default;
  MoveOnlyView& operator=(MoveOnlyView&&)      = default;
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* end() const
  {
    return buffer_ + size_;
  }
};

static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while))>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), int>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), Pred>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), int, Pred>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), int (&)[2], Pred>);
static_assert(!cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), Foo (&)[2], Pred>);
static_assert(cuda::std::is_invocable_v<decltype((cuda::std::views::drop_while)), MoveOnlyView, Pred>);

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

static_assert(!CanBePiped<MoveOnlyView, decltype(cuda::std::views::drop_while)>);
static_assert(CanBePiped<MoveOnlyView, decltype(cuda::std::views::drop_while(Pred{}))>);
static_assert(!CanBePiped<int, decltype(cuda::std::views::drop_while(Pred{}))>);
static_assert(CanBePiped<int (&)[2], decltype(cuda::std::views::drop_while(Pred{}))>);
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_ICC) // template instantiation resulted in
                                                                             // unexpected function type
static_assert(!CanBePiped<Foo (&)[2], decltype(cuda::std::views::drop_while(Pred{}))>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_ICC

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
  int buff[] = {1, 2, 3, 4, 3, 2, 1};

  // Test `views::drop_while(p)(v)`
  {
    using Result          = cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = cuda::std::views::drop_while(Pred{})(MoveOnlyView{buff});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 4, 3, 2, 1};
    assert(equal(result, expected));
  }
  {
    auto const partial    = cuda::std::views::drop_while(Pred{});
    using Result          = cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = partial(MoveOnlyView{buff});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 4, 3, 2, 1};
    assert(equal(result, expected));
  }

  // Test `v | views::drop_while(p)`
  {
    using Result          = cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = MoveOnlyView{buff} | cuda::std::views::drop_while(Pred{});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 4, 3, 2, 1};
    assert(equal(result, expected));
  }
  {
    auto const partial    = cuda::std::views::drop_while(Pred{});
    using Result          = cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = MoveOnlyView{buff} | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 4, 3, 2, 1};
    assert(equal(result, expected));
  }

  // Test `views::drop_while(v, p)`
  {
    using Result          = cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>;
    decltype(auto) result = cuda::std::views::drop_while(MoveOnlyView{buff}, Pred{});
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {3, 4, 3, 2, 1};
    assert(equal(result, expected));
  }

  // Test adaptor | adaptor
  {
    struct Pred2
    {
      __host__ __device__ constexpr bool operator()(int i) const
      {
        return i < 4;
      }
    };
    auto const partial = cuda::std::views::drop_while(Pred{}) | cuda::std::views::drop_while(Pred2{});
    using Result = cuda::std::ranges::drop_while_view<cuda::std::ranges::drop_while_view<MoveOnlyView, Pred>, Pred2>;
    decltype(auto) result = MoveOnlyView{buff} | partial;
    static_assert(cuda::std::same_as<decltype(result), Result>);
    auto expected = {4, 3, 2, 1};
    assert(equal(result, expected));
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
