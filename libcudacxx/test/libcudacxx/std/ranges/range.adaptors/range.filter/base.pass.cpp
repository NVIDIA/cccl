//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr View base() const& requires copy_constructible<View>;
// constexpr View base() &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Range : cuda::std::ranges::view_base
{
  TEST_FUNC constexpr Range()
      : Range{nullptr, nullptr}
  {}

  TEST_FUNC constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  TEST_FUNC constexpr Range(Range const& other)
      : begin_(other.begin_)
      , end_(other.end_)
      , wasCopyInitialized(true)
  {}
  TEST_FUNC constexpr Range(Range&& other)
      : begin_(other.begin_)
      , end_(other.end_)
      , wasMoveInitialized(true)
  {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  [[nodiscard]] TEST_FUNC constexpr int* begin() const
  {
    return begin_;
  }
  [[nodiscard]] TEST_FUNC constexpr int* end() const
  {
    return end_;
  }

  int* begin_;
  int* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

struct Pred
{
  [[nodiscard]] TEST_FUNC bool operator()(int) const
  {
    return true;
  }
};

struct NoCopyRange : cuda::std::ranges::view_base
{
  NoCopyRange() = default;
  TEST_FUNC constexpr explicit NoCopyRange(int*, int*) {}

  NoCopyRange(NoCopyRange const&)            = delete;
  NoCopyRange(NoCopyRange&&)                 = default;
  NoCopyRange& operator=(NoCopyRange const&) = default;
  NoCopyRange& operator=(NoCopyRange&&)      = default;

  [[nodiscard]] TEST_FUNC int* begin() const
  {
    return nullptr;
  }
  [[nodiscard]] TEST_FUNC int* end() const
  {
    return nullptr;
  }
};

template <typename T>
_CCCL_CONCEPT can_call_base_on = _CCCL_REQUIRES_EXPR((T), T t)(cuda::std::forward<T>(t).base());

// Ensure the const& overload is not considered when the base is not copy-constructible
static_assert(!can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred> const&>);
static_assert(!can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>&>);
static_assert(can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>&&>);
static_assert(can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>>);

TEST_FUNC constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the const& overload
  {
    Range range(buff, buff + 8);
    cuda::std::ranges::filter_view<Range, Pred> const view(range, Pred{});
    decltype(auto) result = view.base();
    static_assert(cuda::std::same_as<decltype(result), Range>);
    assert(result.wasCopyInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
  }

  // Check the && overload
  {
    Range range(buff, buff + 8);
    cuda::std::ranges::filter_view<Range, Pred> view(range, Pred{});
    decltype(auto) result = cuda::std::move(view).base();
    static_assert(cuda::std::same_as<decltype(result), Range>);
    assert(result.wasMoveInitialized);
    assert(result.begin() == buff);
    assert(result.end() == buff + 8);
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
