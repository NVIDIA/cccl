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

// constexpr filter_view(View, Pred);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"
#include "types.h"

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr explicit Range(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr int* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return end_;
  }

private:
  int* begin_;
  int* end_;
};

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i % 2 != 0;
  }
};

struct TrackingPred : TrackInitialization
{
  using TrackInitialization::TrackInitialization;
  __host__ __device__ constexpr bool operator()(int) const
  {
    return true;
  }
};

struct TrackingRange
    : TrackInitialization
    , cuda::std::ranges::view_base
{
  using TrackInitialization::TrackInitialization;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

__host__ __device__ constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test explicit syntax
  {
    Range range(buff, buff + 8);
    Pred pred{};
    cuda::std::ranges::filter_view<Range, Pred> view(range, pred);
    auto it = view.begin(), end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 3);
    assert(*it++ == 5);
    assert(*it++ == 7);
    assert(it == end);
  }

  // Test implicit syntax
  {
    Range range(buff, buff + 8);
    Pred pred{};
    cuda::std::ranges::filter_view<Range, Pred> view = {range, pred};
    auto it = view.begin(), end = view.end();
    assert(*it++ == 1);
    assert(*it++ == 3);
    assert(*it++ == 5);
    assert(*it++ == 7);
    assert(it == end);
  }

  // Make sure we move the view
  {
    bool moved = false, copied = false;
    TrackingRange range(&moved, &copied);
    Pred pred{};
    cuda::std::ranges::filter_view<TrackingRange, Pred> view(cuda::std::move(range), pred);
    assert(moved);
    assert(!copied);
    unused(view);
  }

  // Make sure we move the predicate
  {
    bool moved = false, copied = false;
    Range range(buff, buff + 8);
    TrackingPred pred(&moved, &copied);
    cuda::std::ranges::filter_view<Range, TrackingPred> view(range, cuda::std::move(pred));
    assert(moved);
    assert(!copied);
    unused(view);
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
