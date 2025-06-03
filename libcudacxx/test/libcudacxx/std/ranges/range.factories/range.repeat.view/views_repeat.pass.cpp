//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class T>
// views::repeat(T &&) requires constructible_from<ranges::repeat_view<T>, T>;

// template <class T, class Bound>
// views::repeat(T &&, Bound &&) requires constructible_from<ranges::repeat_view<T, Bound>, T, Bound>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "MoveOnly.h"
#include "test_macros.h"

struct NonCopyable
{
  __host__ __device__ NonCopyable(NonCopyable&) = delete;
};

struct NonDefaultCtor
{
  __host__ __device__ NonDefaultCtor(int) {}
};

struct Empty
{};

struct LessThan3
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i < 3;
  }
};

struct EqualTo33
{
  __host__ __device__ constexpr bool operator()(int i) const
  {
    return i == 33;
  }
};

struct Add3
{
  __host__ __device__ constexpr int operator()(int i) const
  {
    return i + 3;
  }
};

// Tp is_object
static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), int>);
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), void>);

// _Bound is semiregular, integer like or cuda::std::unreachable_sentinel_t
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), int, Empty>);
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), int, NonCopyable>);
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), int, NonDefaultCtor>);
static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), int, cuda::std::unreachable_sentinel_t>);

// Tp is copy_constructible
static_assert(!cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), NonCopyable>);

// Tp is move_constructible
static_assert(cuda::std::is_invocable_v<decltype(cuda::std::views::repeat), MoveOnly>);

__host__ __device__ constexpr bool test()
{
  assert(*cuda::std::views::repeat(33).begin() == 33);
  assert(*cuda::std::views::repeat(33, 10).begin() == 33);
  static_assert(cuda::std::same_as<decltype(cuda::std::views::repeat(42)), cuda::std::ranges::repeat_view<int>>);
  static_assert(
    cuda::std::same_as<decltype(cuda::std::views::repeat(42, 3)), cuda::std::ranges::repeat_view<int, int>>);
  static_assert(cuda::std::same_as<decltype(cuda::std::views::repeat), decltype(cuda::std::ranges::views::repeat)>);

#if 0 // Not yet implemented views
  // unbound && drop_view
  {
    auto r = cuda::std::views::repeat(33) | cuda::std::views::drop(3);
    static_assert(!cuda::std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
  }

  // bound && drop_view
  {
    auto r = cuda::std::views::repeat(33, 8) | cuda::std::views::drop(3);
    static_assert(cuda::std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 5);
  }

  // unbound && take_view
  {
    auto r = cuda::std::views::repeat(33) | cuda::std::views::take(3);
    static_assert(cuda::std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 3);
  }

  // bound && take_view
  {
    auto r = cuda::std::views::repeat(33, 8) | cuda::std::views::take(3);
    static_assert(cuda::std::ranges::sized_range<decltype(r)>);
    assert(*r.begin() == 33);
    assert(r.size() == 3);
  }
#endif // Not yet implemented views

  // bound && transform_view
  {
    auto r = cuda::std::views::repeat(33, 8) | cuda::std::views::transform(Add3{});
    assert(*r.begin() == 36);
    assert(r.size() == 8);
  }

  // unbound && transform_view
  {
    auto r = cuda::std::views::repeat(33) | cuda::std::views::transform(Add3{});
    assert(*r.begin() == 36);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
