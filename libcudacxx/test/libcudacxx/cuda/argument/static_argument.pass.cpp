//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test()
{
  // Basic value
  {
    constexpr auto sa = cuda::static_argument<42>{};
    static_assert(sa.value == 42);
    static_assert(cuda::std::is_same_v<decltype(sa)::value_type, int>);
  }

  // Different types
  {
    constexpr auto sa_long = cuda::static_argument<100L>{};
    static_assert(sa_long.value == 100L);
    static_assert(cuda::std::is_same_v<decltype(sa_long)::value_type, long>);
  }

  // Negative value
  {
    constexpr auto sa_neg = cuda::static_argument<-1>{};
    static_assert(sa_neg.value == -1);
  }

  // Array value (per-segment at compile time)
  {
    constexpr auto sa_arr = cuda::static_argument<cuda::std::array<int, 3>{128, 256, 512}>{};
    static_assert(sa_arr.value[0] == 128);
    static_assert(sa_arr.value[1] == 256);
    static_assert(sa_arr.value[2] == 512);
    static_assert(cuda::std::is_same_v<decltype(sa_arr)::value_type, cuda::std::array<int, 3>>);
  }

  // Bounds: scalar
  {
    constexpr auto sa = cuda::static_argument<42>{};
    static_assert(cuda::argument_lowest(sa) == 42);
    static_assert(cuda::argument_max(sa) == 42);
    static_assert(cuda::argument_lowest(sa) == 42);
    static_assert(cuda::argument_max(sa) == 42);
  }

  // Bounds: array — computes min/max of elements
  {
    constexpr auto sa = cuda::static_argument<cuda::std::array<int, 3>{128, 256, 512}>{};
    static_assert(cuda::argument_lowest(sa) == 128);
    static_assert(cuda::argument_max(sa) == 512);
    static_assert(cuda::argument_lowest(sa) == 128);
    static_assert(cuda::argument_max(sa) == 512);
  }

  // Traits
  {
    using traits = cuda::argument_traits<cuda::static_argument<42>>;
    static_assert(!traits::is_deferred);
    static_assert(cuda::std::is_same_v<traits::value_type, int>);
  }

  // Single value: scalar is single, array is not
  {
    static_assert(cuda::__is_single_value_v<cuda::argument_traits<cuda::static_argument<42>>::value_type>);
    static_assert(!cuda::__is_single_value_v<
                  cuda::argument_traits<cuda::static_argument<cuda::std::array<int, 3>{1, 2, 3}>>::value_type>);
  }

  // Unwrap: scalar
  {
    constexpr auto sa         = cuda::static_argument<42>{};
    constexpr const auto& val = cuda::unwrap_argument(sa);
    static_assert(val == 42);
  }

  // Unwrap: array
  {
    constexpr auto sa         = cuda::static_argument<cuda::std::array<int, 3>{10, 20, 30}>{};
    constexpr const auto& val = cuda::unwrap_argument(sa);
    static_assert(val[0] == 10);
    static_assert(val[2] == 30);
  }
}

int main(int, char**)
{
  test();
  return 0;
}
