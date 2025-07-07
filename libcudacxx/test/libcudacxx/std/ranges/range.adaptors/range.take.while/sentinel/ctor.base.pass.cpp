//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

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
  __host__ __device__ int* begin() const
  {
    return nullptr;
  };
  __host__ __device__ Sent end()
  {
    return Sent{42};
  };
};
static_assert(cuda::std::ranges::range<Range>);

struct Pred
{
  __host__ __device__ bool operator()(int i) const;
};

// Test explicit
template <class T>
__host__ __device__ void conversion_test(T);

template <class T, class... Args>
_CCCL_CONCEPT ImplicitlyConstructible =
  _CCCL_REQUIRES_EXPR((T, variadic Args), Args&&... args)((conversion_test<T>({cuda::std::forward<Args>(args)...})));

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
