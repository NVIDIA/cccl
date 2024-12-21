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

// constexpr auto end();

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"

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

struct CommonRange : cuda::std::ranges::view_base
{
  using Iterator = forward_iterator<int*>;
  __host__ __device__ constexpr explicit CommonRange(int* b, int* e)
      : begin_(b)
      , end_(e)
  {}
  __host__ __device__ constexpr Iterator begin() const
  {
    return Iterator(begin_);
  }
  __host__ __device__ constexpr Iterator end() const
  {
    return Iterator(end_);
  }

private:
  int* begin_;
  int* end_;
};

__host__ __device__ constexpr bool test()
{
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.end()`
  {
    Range range(buff, buff + 1);
    auto pred = [](int) {
      return true;
    };
    cuda::std::ranges::filter_view view(range, pred);
    using FilterSentinel = cuda::std::ranges::sentinel_t<decltype(view)>;
    ASSERT_SAME_TYPE(FilterSentinel, decltype(view.end()));
  }

  // end() on an empty range
  {
    Range range(buff, buff);
    auto pred = [](int) {
      return true;
    };
    cuda::std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff);
  }

  // end() on a 1-element range
  {
    Range range(buff, buff + 1);
    auto pred = [](int) {
      return true;
    };
    cuda::std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff + 1);
    static_assert(!cuda::std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  // end() on a 2-element range
  {
    Range range(buff, buff + 2);
    auto pred = [](int) {
      return true;
    };
    cuda::std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(base(end.base())) == buff + 2);
    static_assert(!cuda::std::is_same_v<decltype(end), decltype(view.begin())>);
  }

  // end() on a N-element range
  {
    for (int k = 1; k != 8; ++k)
    {
      Range range(buff, buff + 8);
      auto pred = [=](int i) {
        return i == k;
      };
      cuda::std::ranges::filter_view view(range, pred);
      auto end = view.end();
      assert(base(base(end.base())) == buff + 8);
      static_assert(!cuda::std::is_same_v<decltype(end), decltype(view.begin())>);
    }
  }

  // end() on a common_range
  {
    CommonRange range(buff, buff + 8);
    auto pred = [](int i) {
      return i % 2 == 0;
    };
    cuda::std::ranges::filter_view view(range, pred);
    auto end = view.end();
    assert(base(end.base()) == buff + 8);
    static_assert(cuda::std::is_same_v<decltype(end), decltype(view.begin())>);
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
