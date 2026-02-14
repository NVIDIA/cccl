//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto size() const requires (!same_as<Bound, unreachable_sentinel_t>);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/ranges>

#include "test_macros.h"

template <class T>
_CCCL_CONCEPT has_size = _CCCL_REQUIRES_EXPR((T), T&& view)((cuda::std::forward<T>(view).size()));

static_assert(has_size<cuda::std::ranges::repeat_view<int, int>>);
static_assert(!has_size<cuda::std::ranges::repeat_view<int>>);
static_assert(!has_size<cuda::std::ranges::repeat_view<int, cuda::std::unreachable_sentinel_t>>);

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::ranges::repeat_view<int, int> rv(10, 20);
    assert(rv.size() == 20);
  }

  {
    constexpr int int_max = cuda::std::numeric_limits<int>::max();
    cuda::std::ranges::repeat_view<int, int> rv(10, int_max);
    assert(rv.size() == int_max);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
