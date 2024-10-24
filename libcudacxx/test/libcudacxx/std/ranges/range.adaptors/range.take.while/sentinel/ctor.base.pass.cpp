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

// constexpr explicit sentinel(sentinel_t<Base> end, const Pred* pred);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "../types.h"

struct Sent
{
  int i;

  __host__ __device__ friend constexpr bool operator==(int* iter, const Sent& s)
  {
    return s.i > *iter;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const Sent& s, int* iter)
  {
    return s.i > *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(int* iter, const Sent& s)
  {
    return s.i <= *iter;
  }
  __host__ __device__ friend constexpr bool operator!=(const Sent& s, int* iter)
  {
    return s.i <= *iter;
  }
#endif // TEST_STD_VER <= 2017
};

struct Range : cuda::std::ranges::view_base
{
  __host__ __device__ int* begin() const;
  __host__ __device__ Sent end();
};
static_assert(cuda::std::ranges::range<Range>);

struct Pred
{
  __host__ __device__ bool operator()(int i) const;
};

// Test explicit
template <class T>
__host__ __device__ void conversion_test(T);

#if TEST_STD_VER >= 2020
template <class T, class... Args>
concept ImplicitlyConstructible = requires(Args&&... args) { conversion_test<T>({cuda::std::forward<Args>(args)...}); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(ImplicitlyConstructible_,
                             requires(Args&&... args)((conversion_test<T>({cuda::std::forward<Args>(args)...}))));

template <class T, class... Args>
_LIBCUDACXX_CONCEPT ImplicitlyConstructible = _LIBCUDACXX_FRAGMENT(ImplicitlyConstructible_, T, Args...);
#endif // TEST_STD_VER <= 2017

static_assert(ImplicitlyConstructible<int, int>);

static_assert(
  cuda::std::is_constructible_v<cuda::std::ranges::sentinel_t<cuda::std::ranges::take_while_view<Range, Pred>>,
                                cuda::std::ranges::sentinel_t<Range>,
                                const Pred*>);
static_assert(!ImplicitlyConstructible<cuda::std::ranges::sentinel_t<cuda::std::ranges::take_while_view<Range, Pred>>,
                                       cuda::std::ranges::sentinel_t<Range>,
                                       const Pred*>);

__host__ __device__ constexpr bool test()
{
  // base is init correctly
  {
    using R        = cuda::std::ranges::take_while_view<Range, bool (*)(int)>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    Sentinel s1(Sent{5}, nullptr);
    assert(s1.base().i == 5);
  }

  // pred is init correctly
  {
    bool called = false;
    auto pred   = [&](int) {
      called = true;
      return false;
    };

    using R        = cuda::std::ranges::take_while_view<Range, decltype(pred)>;
    using Sentinel = cuda::std::ranges::sentinel_t<R>;

    int i     = 10;
    int* iter = &i;
    Sentinel s(Sent{0}, &pred);

    bool b = iter == s;
    assert(called);
    assert(b);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
