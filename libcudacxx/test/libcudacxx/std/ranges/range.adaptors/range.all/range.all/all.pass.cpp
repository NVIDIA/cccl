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

// cuda::std::views::all;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

__device__ int globalBuff[8];

template <bool IsNoexcept>
struct View : cuda::std::ranges::view_base
{
  int start_ = 0;
  __host__ __device__ constexpr explicit View() noexcept(IsNoexcept){};
  __host__ __device__ constexpr explicit View(int start)
      : start_(start)
  {}
  __host__ __device__ constexpr View(View&& other) noexcept(IsNoexcept)
      : start_(cuda::std::exchange(other.start_, 0))
  {}
  __host__ __device__ constexpr View& operator=(View&& other) noexcept(IsNoexcept)
  {
    start_ = cuda::std::exchange(other.start_, 0);
    return *this;
  }
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
static_assert(cuda::std::ranges::view<View<true>>);
static_assert(cuda::std::ranges::view<View<false>>);

static_assert(cuda::std::ranges::range<View<false>>);
static_assert(cuda::std::movable<View<false>>);
static_assert(cuda::std::ranges::enable_view<View<false>>);

template <bool IsNoexcept>
struct CopyableView : cuda::std::ranges::view_base
{
  int start_ = 0;
  __host__ __device__ constexpr explicit CopyableView() noexcept(IsNoexcept){};
  __host__ __device__ constexpr CopyableView(CopyableView const& other) noexcept(IsNoexcept)
      : start_(other.start_)
  {}
  __host__ __device__ constexpr CopyableView& operator=(CopyableView const& other) noexcept(IsNoexcept)
  {
    start_ = other.start_;
    return *this;
  }
  __host__ __device__ constexpr explicit CopyableView(int start) noexcept
      : start_(start)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
static_assert(cuda::std::ranges::view<CopyableView<true>>);
static_assert(cuda::std::ranges::view<CopyableView<false>>);

struct MoveOnlyView : cuda::std::ranges::view_base
{
  MoveOnlyView()                               = default;
  MoveOnlyView(const MoveOnlyView&)            = delete;
  MoveOnlyView& operator=(const MoveOnlyView&) = delete;
  MoveOnlyView(MoveOnlyView&&)                 = default;
  MoveOnlyView& operator=(MoveOnlyView&&)      = default;

  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct Range
{
  int start_;
  __host__ __device__ constexpr explicit Range(int start) noexcept
      : start_(start)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};

struct BorrowableRange
{
  int start_;
  __host__ __device__ constexpr explicit BorrowableRange(int start) noexcept
      : start_(start)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return globalBuff + start_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return globalBuff + 8;
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<BorrowableRange> = true;

struct RandomAccessRange
{
  __host__ __device__ constexpr auto begin()
  {
    return random_access_iterator<int*>(globalBuff);
  }
  __host__ __device__ constexpr auto end()
  {
    return sized_sentinel(random_access_iterator<int*>(globalBuff + 8));
  }
};
template <>
inline constexpr bool cuda::std::ranges::enable_borrowed_range<RandomAccessRange> = true;

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

__host__ __device__ constexpr bool test()
{
  {
    ASSERT_SAME_TYPE(decltype(cuda::std::views::all(View<true>())), View<true>);
    static_assert(noexcept(cuda::std::views::all(View<true>())));
// old GCC seems to fall over the noexcept clauses here
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9) && !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_ICC)
    static_assert(!noexcept(cuda::std::views::all(View<false>())));
#endif // no broken noexcept

    auto viewCopy = cuda::std::views::all(View<true>(2));
    ASSERT_SAME_TYPE(decltype(viewCopy), View<true>);
    assert(cuda::std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(cuda::std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    ASSERT_SAME_TYPE(decltype(cuda::std::views::all(cuda::std::declval<const CopyableView<true>&>())),
                     CopyableView<true>);
    static_assert(noexcept(cuda::std::views::all(CopyableView<true>())));
#if (!defined(TEST_COMPILER_GCC) || __GNUC__ >= 9) && !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_ICC)
    static_assert(!noexcept(cuda::std::views::all(CopyableView<false>())));
#endif // no broken noexcept

    CopyableView<true> view(2);
    auto viewCopy = cuda::std::views::all(view);
    ASSERT_SAME_TYPE(decltype(viewCopy), CopyableView<true>);
    assert(cuda::std::ranges::begin(viewCopy) == globalBuff + 2);
    assert(cuda::std::ranges::end(viewCopy) == globalBuff + 8);
  }

  {
    Range range(2);
    auto ref = cuda::std::views::all(range);
    ASSERT_SAME_TYPE(decltype(ref), cuda::std::ranges::ref_view<Range>);
    assert(cuda::std::ranges::begin(ref) == globalBuff + 2);
    assert(cuda::std::ranges::end(ref) == globalBuff + 8);

    auto own = cuda::std::views::all(cuda::std::move(range));
    ASSERT_SAME_TYPE(decltype(own), cuda::std::ranges::owning_view<Range>);
    assert(cuda::std::ranges::begin(own) == globalBuff + 2);
    assert(cuda::std::ranges::end(own) == globalBuff + 8);

    auto cref = cuda::std::views::all(cuda::std::as_const(range));
    ASSERT_SAME_TYPE(decltype(cref), cuda::std::ranges::ref_view<const Range>);
    assert(cuda::std::ranges::begin(cref) == globalBuff + 2);
    assert(cuda::std::ranges::end(cref) == globalBuff + 8);

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_GCC)
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::all), const Range&&>);
#endif // !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_GCC)
  }

  {
    auto own = cuda::std::views::all(BorrowableRange(2));
    ASSERT_SAME_TYPE(decltype(own), cuda::std::ranges::owning_view<BorrowableRange>);
    assert(cuda::std::ranges::begin(own) == globalBuff + 2);
    assert(cuda::std::ranges::end(own) == globalBuff + 8);
  }

  {
    auto own = cuda::std::views::all(RandomAccessRange());
    ASSERT_SAME_TYPE(decltype(own), cuda::std::ranges::owning_view<RandomAccessRange>);
    assert(base(cuda::std::ranges::begin(own)) == globalBuff);
    assert(base(base(cuda::std::ranges::end(own))) == globalBuff + 8);
  }

  // Check SFINAE friendliness of the call operator
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_GCC)
  {
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::all)>);
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::all), RandomAccessRange, RandomAccessRange>);

    // `views::all(v)` is expression equivalent to `decay-copy(v)` if the decayed type
    // of `v` models `view`. If `v` is an lvalue-reference to a move-only view, the
    // expression should be ill-formed because `v` is not copyable
    static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::all), MoveOnlyView&>);
  }
#endif // !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_GCC)

  // Test that cuda::std::views::all is a range adaptor
  { // Test `v | views::all`
    {
      Range range(0);
      auto result = range | cuda::std::views::all;
      ASSERT_SAME_TYPE(decltype(result), cuda::std::ranges::ref_view<Range>);
      assert(&result.base() == &range);
    }

    // Test `adaptor | views::all`
    {
      Range range(0);
      auto f = [](int i) {
        return i;
      };
      auto const partial    = cuda::std::views::transform(f) | cuda::std::views::all;
      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::ref_view<Range>, decltype(f)>;
      decltype(auto) result = partial(range);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(&result.base().base() == &range);
    }

    // Test `views::all | adaptor`
    {
      Range range(0);
      auto f = [](int i) {
        return i;
      };
      auto const partial    = cuda::std::views::all | cuda::std::views::transform(f);
      using Result          = cuda::std::ranges::transform_view<cuda::std::ranges::ref_view<Range>, decltype(f)>;
      decltype(auto) result = partial(range);
      static_assert(cuda::std::same_as<decltype(result), Result>);
      assert(&result.base().base() == &range);
    }

    {
      struct NotAView
      {};
      static_assert(CanBePiped<Range&, decltype(cuda::std::views::all)>);
      static_assert(!CanBePiped<NotAView, decltype(cuda::std::views::all)>);
    }
  }

  {
    static_assert(cuda::std::same_as<decltype(cuda::std::views::all), decltype(cuda::std::ranges::views::all)>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_ADDRESSOF

  return 0;
}
