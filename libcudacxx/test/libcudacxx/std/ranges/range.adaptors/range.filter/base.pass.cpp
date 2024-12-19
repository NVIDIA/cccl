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

// constexpr View base() const& requires copy_constructible<View>;
// constexpr View base() &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr Range(Range const& other)
      : begin_(other.begin_)
      , end_(other.end_)
      , wasCopyInitialized(true)
  {}
  __host__ __device__ constexpr Range(Range&& other)
      : begin_(other.begin_)
      , end_(other.end_)
      , wasMoveInitialized(true)
  {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int* end() const
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
  __host__ __device__ bool operator()(int) const;
};

struct NoCopyRange : cuda::std::ranges::view_base
{
  __host__ __device__ explicit NoCopyRange(int*, int*);
  NoCopyRange(NoCopyRange const&)            = delete;
  NoCopyRange(NoCopyRange&&)                 = default;
  NoCopyRange& operator=(NoCopyRange const&) = default;
  NoCopyRange& operator=(NoCopyRange&&)      = default;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

#if TEST_STD_VER >= 2020
template <typename T>
concept can_call_base_on = requires(T t) { cuda::std::forward<T>(t).base(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool can_call_base_on = false;

template <class T>
inline constexpr bool can_call_base_on<T, cuda::std::void_t<decltype(cuda::std::declval<T>().base())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
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

  // Ensure the const& overload is not considered when the base is not copy-constructible
  {
    static_assert(!can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred> const&>);
    static_assert(!can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>&>);
    static_assert(can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>&&>);
    static_assert(can_call_base_on<cuda::std::ranges::filter_view<NoCopyRange, Pred>>);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
