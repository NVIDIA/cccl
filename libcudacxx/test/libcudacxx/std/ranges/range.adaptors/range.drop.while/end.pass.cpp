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
#include <cuda/std/ranges>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"

struct View : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};

// Test that end is not const
#if TEST_STD_VER >= 2020
template <class T>
concept HasEnd = requires(T t) { t.end(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasEnd = false;

template <class T>
inline constexpr bool HasEnd<T, cuda::std::void_t<decltype(cuda::std::declval<T>().end())>> = true;
#endif // TEST_STD_VER <= 2017

struct Pred
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
};

static_assert(HasEnd<cuda::std::ranges::drop_while_view<View, Pred>>);
static_assert(!HasEnd<const cuda::std::ranges::drop_while_view<View, Pred>>);

__host__ __device__ constexpr bool test()
{
  // return iterator
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    cuda::std::ranges::drop_while_view dwv(buffer, Pred{});
    decltype(auto) st = dwv.end();
    static_assert(cuda::std::same_as<decltype(st), int*>);
    assert(st == buffer + 11);
  }

  // return sentinel
  {
    using Iter   = int*;
    using Sent   = sentinel_wrapper<Iter>;
    using Range  = cuda::std::ranges::subrange<Iter, Sent>;
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    Range range  = {buffer, Sent{buffer + 11}};
    cuda::std::ranges::drop_while_view dwv(range, Pred{});
    decltype(auto) st = dwv.end();
    static_assert(cuda::std::same_as<decltype(st), Sent>);
    assert(base(st) == buffer + 11);
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
