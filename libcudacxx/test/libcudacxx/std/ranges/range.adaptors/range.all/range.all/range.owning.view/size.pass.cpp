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

// constexpr auto size() requires sized_range<R>
// constexpr auto size() const requires sized_range<const R>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasSize = requires(T t) { t.size(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool HasSize = false;

template <class T>
constexpr bool HasSize<T, cuda::std::void_t<decltype(cuda::std::declval<T>().size())>> = true;
#endif // TEST_STD_VER <= 2017

struct SubtractableIters
{
  __host__ __device__ forward_iterator<int*> begin();
  __host__ __device__ sized_sentinel<forward_iterator<int*>> end();
};

struct NoSize
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<int*> end();
};

struct SizeMember
{
  __host__ __device__ bidirectional_iterator<int*> begin();
  __host__ __device__ bidirectional_iterator<int*> end();
  __host__ __device__ int size() const;
};

__host__ __device__ constexpr bool test()
{
  {
    using OwningView = cuda::std::ranges::owning_view<SubtractableIters>;
    static_assert(cuda::std::ranges::sized_range<OwningView&>);
    static_assert(!cuda::std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasSize<OwningView&>);
    static_assert(HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>);
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    using OwningView = cuda::std::ranges::owning_view<NoSize>;
    static_assert(!HasSize<OwningView&>);
    static_assert(!HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>);
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    using OwningView = cuda::std::ranges::owning_view<SizeMember>;
    static_assert(cuda::std::ranges::sized_range<OwningView&>);
    static_assert(!cuda::std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasSize<OwningView&>);
    static_assert(HasSize<OwningView&&>);
    static_assert(!HasSize<const OwningView&>); // not a range, therefore no size()
    static_assert(!HasSize<const OwningView&&>);
  }
  {
    // Test an empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a));
    assert(ov.size() == 0);
    assert(cuda::std::as_const(ov).size() == 0);
  }
  {
    // Test a non-empty view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a + 1));
    assert(ov.size() == 1);
    assert(cuda::std::as_const(ov).size() == 1);
  }
  {
    // Test a non-view.
    cuda::std::array<int, 2> a = {1, 2};
    auto ov                    = cuda::std::ranges::owning_view(cuda::std::move(a));
    assert(ov.size() == 2);
    assert(cuda::std::as_const(ov).size() == 2);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
