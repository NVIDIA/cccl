//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// filter_view() requires cuda::std::default_initializable<View> &&
//                        cuda::std::default_initializable<Pred> = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr DefaultConstructibleView()
      : begin_(buff)
      , end_(buff + 8)
  {}
  __host__ __device__ constexpr int const* begin() const
  {
    return begin_;
  }
  __host__ __device__ constexpr int const* end() const
  {
    return end_;
  }

private:
  int const* begin_;
  int const* end_;
};

struct DefaultConstructiblePredicate
{
  DefaultConstructiblePredicate() = default;
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i % 2 == 0;
  }
};

struct NoDefaultView : cuda::std::ranges::view_base
{
  __host__ __device__ NoDefaultView() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

struct NoDefaultPredicate
{
  __host__ __device__ NoDefaultPredicate() = delete;
  __host__ __device__ constexpr bool operator()(int) const
  {
    return true;
  }
};

struct NoexceptView : cuda::std::ranges::view_base
{
  __host__ __device__ NoexceptView() noexcept;
  __host__ __device__ int const* begin() const;
  __host__ __device__ int const* end() const;
};

struct NoexceptPredicate
{
  __host__ __device__ NoexceptPredicate() noexcept;
  __host__ __device__ bool operator()(int) const;
};

__host__ __device__ constexpr bool test()
{
  {
    using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
    View view;
    auto it = view.begin(), end = view.end();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it++ == 6);
    assert(*it++ == 8);
    assert(it == end);
  }

  {
    using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
    View view  = {};
    auto it = view.begin(), end = view.end();
    assert(*it++ == 2);
    assert(*it++ == 4);
    assert(*it++ == 6);
    assert(*it++ == 8);
    assert(it == end);
  }

  // Check cases where the default constructor isn't provided
  {
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::filter_view<NoDefaultView, DefaultConstructiblePredicate>>);
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::std::ranges::filter_view<DefaultConstructibleView, NoDefaultPredicate>>);
    static_assert(
      !cuda::std::is_default_constructible_v<cuda::std::ranges::filter_view<NoDefaultView, NoDefaultPredicate>>);
  }

  // Check noexcept-ness
  {
#if !TEST_COMPILER(GCC, <, 9) // broken noexcept
#  if TEST_STD_VER <= 2017 // This assert always triggers in c++20 mode
    {
      using View = cuda::std::ranges::filter_view<DefaultConstructibleView, DefaultConstructiblePredicate>;
      static_assert(!noexcept(View()));
    }
#  endif // TEST_STD_VER <= 2017
#endif // no broken noexcept
    {
      using View = cuda::std::ranges::filter_view<NoexceptView, NoexceptPredicate>;
      static_assert(noexcept(View()));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020 && defined(_CCCL_BUILTIN_ADDRESSOF)

  return 0;
}
