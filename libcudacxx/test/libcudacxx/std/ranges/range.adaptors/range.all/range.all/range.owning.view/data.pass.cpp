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

// constexpr auto data() requires contiguous_range<R>
// constexpr auto data() const requires contiguous_range<const R>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasData = requires(T t) { t.data(); };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
constexpr bool HasData = false;

template <class T>
constexpr bool HasData<T, cuda::std::void_t<decltype(cuda::std::declval<T>().data())>> = true;
#endif // TEST_STD_VER <= 2017

struct ContiguousIters
{
  __host__ __device__ contiguous_iterator<int*> begin();
  __host__ __device__ sentinel_wrapper<contiguous_iterator<int*>> end();
};

struct NoData
{
  __host__ __device__ random_access_iterator<int*> begin();
  __host__ __device__ random_access_iterator<int*> end();
};

__host__ __device__ constexpr bool test()
{
  {
    using OwningView = cuda::std::ranges::owning_view<ContiguousIters>;
    static_assert(cuda::std::ranges::contiguous_range<OwningView&>);
    static_assert(!cuda::std::ranges::range<const OwningView&>); // no begin/end
    static_assert(HasData<OwningView&>);
    static_assert(HasData<OwningView&&>);
    static_assert(!HasData<const OwningView&>);
    static_assert(!HasData<const OwningView&&>);
  }
  {
    using OwningView = cuda::std::ranges::owning_view<NoData>;
    static_assert(!HasData<OwningView&>);
    static_assert(!HasData<OwningView&&>);
    static_assert(!HasData<const OwningView&>);
    static_assert(!HasData<const OwningView&&>);
  }
  {
    // Test a view.
    int a[] = {1};
    auto ov = cuda::std::ranges::owning_view(cuda::std::ranges::subrange(a, a + 1));
    assert(ov.data() == a);
    assert(cuda::std::as_const(ov).data() == a);
  }
  {
    // Test a non-view.
    cuda::std::array<int, 2> a = {1, 2};
    auto ov                    = cuda::std::ranges::owning_view(cuda::std::move(a));
    assert(ov.data() != a.data()); // because it points into the copy
    assert(cuda::std::as_const(ov).data() != a.data());
  }
  return true;
}

int main(int, char**)
{
  test();
#if 0 // note #2751-D: access to uninitialized object
  static_assert(test());
#endif

  return 0;
}
