//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class W, class Bound>
//     requires (!is-integer-like<W> || !is-integer-like<Bound> ||
//               (is-signed-integer-like<W> == is-signed-integer-like<Bound>))
//     iota_view(W, Bound) -> iota_view<W, Bound>;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

template <class T, class U>
_CCCL_CONCEPT CanDeduce = _CCCL_REQUIRES_EXPR((T, U), const T& t, const U& u)(cuda::std::ranges::iota_view(t, u));

__host__ __device__ void test()
{
  static_assert(
    cuda::std::same_as<decltype(cuda::std::ranges::iota_view(0, 0)), cuda::std::ranges::iota_view<int, int>>);

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::iota_view(0)),
                                   cuda::std::ranges::iota_view<int, cuda::std::unreachable_sentinel_t>>);

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::iota_view(0, cuda::std::unreachable_sentinel)),
                                   cuda::std::ranges::iota_view<int, cuda::std::unreachable_sentinel_t>>);

  static_assert(cuda::std::same_as<decltype(cuda::std::ranges::iota_view(0, IntComparableWith(0))),
                                   cuda::std::ranges::iota_view<int, IntComparableWith<int>>>);

  static_assert(CanDeduce<int, int>);
  static_assert(!CanDeduce<int, unsigned>);
  static_assert(!CanDeduce<unsigned, int>);
}

int main(int, char**)
{
  return 0;
}
