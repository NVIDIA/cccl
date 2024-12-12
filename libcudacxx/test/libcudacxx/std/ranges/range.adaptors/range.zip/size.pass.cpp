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

// constexpr auto size() requires(sized_range<Views>&&...)
// constexpr auto size() const requires(sized_range<const Views>&&...)

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "types.h"

STATIC_TEST_GLOBAL_VAR int buffer[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

struct View : cuda::std::ranges::view_base
{
  cuda::std::size_t size_ = 0;
  __host__ __device__ constexpr View(cuda::std::size_t s)
      : size_(s)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return buffer;
  }
  __host__ __device__ constexpr auto end() const
  {
    return buffer + size_;
  }
};

struct SizedNonConst : cuda::std::ranges::view_base
{
  using iterator          = forward_iterator<int*>;
  cuda::std::size_t size_ = 0;
  __host__ __device__ constexpr SizedNonConst(cuda::std::size_t s)
      : size_(s)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return iterator{buffer};
  }
  __host__ __device__ constexpr auto end() const
  {
    return iterator{buffer + size_};
  }
  __host__ __device__ constexpr cuda::std::size_t size()
  {
    return size_;
  }
};

struct StrangeSizeView : cuda::std::ranges::view_base
{
  __host__ __device__ constexpr auto begin() const
  {
    return buffer;
  }
  __host__ __device__ constexpr auto end() const
  {
    return buffer + 8;
  }

  __host__ __device__ constexpr auto size()
  {
    return 5;
  }
  __host__ __device__ constexpr auto size() const
  {
    return 6;
  }
};

__host__ __device__ constexpr bool test()
{
  {
    // single range
    cuda::std::ranges::zip_view v(View(8));
    assert(v.size() == 8);
    assert(cuda::std::as_const(v).size() == 8);
  }

  {
    // multiple ranges same type
    cuda::std::ranges::zip_view v(View(2), View(3));
    assert(v.size() == 2);
    assert(cuda::std::as_const(v).size() == 2);
  }

  {
    // multiple ranges different types
    cuda::std::ranges::zip_view v(cuda::std::views::iota(0, 500), View(3));
    assert(v.size() == 3);
    assert(cuda::std::as_const(v).size() == 3);
  }

  {
    // const-view non-sized range
    cuda::std::ranges::zip_view v(SizedNonConst(2), View(3));
    assert(v.size() == 2);
    static_assert(cuda::std::ranges::sized_range<decltype(v)>);
    static_assert(!cuda::std::ranges::sized_range<decltype(cuda::std::as_const(v))>);
  }

  {
    // const/non-const has different sizes
    cuda::std::ranges::zip_view v(StrangeSizeView{});
    assert(v.size() == 5);
    assert(cuda::std::as_const(v).size() == 6);
  }

  {
    // underlying range not sized
    cuda::std::ranges::zip_view v(InputCommonView{buffer});
    static_assert(!cuda::std::ranges::sized_range<decltype(v)>);
    static_assert(!cuda::std::ranges::sized_range<decltype(cuda::std::as_const(v))>);
    unused(v);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
