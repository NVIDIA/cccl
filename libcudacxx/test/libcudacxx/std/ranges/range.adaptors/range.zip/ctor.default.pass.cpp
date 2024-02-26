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

// zip_view() = default;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int buff[] = {1, 2, 3};

struct DefaultConstructibleView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr DefaultConstructibleView()
      : begin_(buff)
      , end_(buff + 3)
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

struct NoDefaultCtrView : cuda::std::ranges::view_base
{
  NoDefaultCtrView() = delete;
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

// The default constructor requires all underlying views to be default constructible.
// It is implicitly required by the tuple's constructor. If any of the iterators are
// not default constructible, zip iterator's =default would be implicitly deleted.
static_assert(cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<DefaultConstructibleView>>);
static_assert(cuda::std::is_default_constructible_v<
              cuda::std::ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>>);
static_assert(
  !cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<DefaultConstructibleView, NoDefaultCtrView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<NoDefaultCtrView, NoDefaultCtrView>>);
static_assert(!cuda::std::is_default_constructible_v<cuda::std::ranges::zip_view<NoDefaultCtrView>>);

__host__ __device__ constexpr bool test()
{
  {
    using View = cuda::std::ranges::zip_view<DefaultConstructibleView, DefaultConstructibleView>;
    View v     = View(); // the default constructor is not explicit
    assert(v.size() == 3);
    auto it    = v.begin();
    using Pair = cuda::std::pair<const int&, const int&>;
    assert(*it++ == Pair(buff[0], buff[0]));
    assert(*it++ == Pair(buff[1], buff[1]));
    assert(*it == Pair(buff[2], buff[2]));
  }

  return true;
}

int main(int, char**)
{
  test();
#ifndef TEST_COMPILER_CUDACC_BELOW_11_3
  static_assert(test(), "");
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3

  return 0;
}
