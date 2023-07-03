//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class W, class Bound>
//     requires (!is-integer-like<W> || !is-integer-like<Bound> ||
//               (is-signed-integer-like<W> == is-signed-integer-like<Bound>))
//     iota_view(W, Bound) -> iota_view<W, Bound>;

#include <cuda/std/ranges>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER > 17
template<class T, class U>
concept CanDeduce = requires(const T& t, const U& u) {
  cuda::std::ranges::iota_view(t, u);
};
#else
template<class T, class U, class = void>
inline constexpr bool CanDeduce = false;

template<class T, class U>
inline constexpr bool CanDeduce<T, U,
  cuda::std::void_t<decltype(cuda::std::ranges::iota_view(cuda::std::declval<const T&>(), cuda::std::declval<const U&>()))>> = true;
#endif

__host__ __device__ void test() {
  static_assert(cuda::std::same_as<
    decltype(cuda::std::ranges::iota_view(0, 0)),
    cuda::std::ranges::iota_view<int, int>
  >);

  static_assert(cuda::std::same_as<
    decltype(cuda::std::ranges::iota_view(0)),
    cuda::std::ranges::iota_view<int, cuda::std::unreachable_sentinel_t>
  >);

  static_assert(cuda::std::same_as<
    decltype(cuda::std::ranges::iota_view(0, cuda::std::unreachable_sentinel)),
    cuda::std::ranges::iota_view<int, cuda::std::unreachable_sentinel_t>
  >);

  static_assert(cuda::std::same_as<
    decltype(cuda::std::ranges::iota_view(0, IntComparableWith(0))),
    cuda::std::ranges::iota_view<int, IntComparableWith<int>>
  >);

#ifndef TEST_COMPILER_NVCC_BELOW_11_3
  static_assert( CanDeduce<int, int>);
#endif // !TEST_COMPILER_NVCC_BELOW_11_3
  static_assert(!CanDeduce<int, unsigned>);
  static_assert(!CanDeduce<unsigned, int>);
}

int main(int, char**) {
  return 0;
}
