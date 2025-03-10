//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto data() requires contiguous_range<R>
// constexpr auto data() const requires contiguous_range<const R>

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

template <class T>
_CCCL_CONCEPT HasData = _CCCL_REQUIRES_EXPR((T), T t)(unused(t.data()));

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
  static_assert(test());

  return 0;
}
