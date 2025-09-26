//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class day;

// constexpr bool operator==(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} == unsigned{y}.
// constexpr strong_ordering operator<=>(const day& x, const day& y) noexcept;
//   Returns: unsigned{x} <=> unsigned{y}.

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using day = cuda::std::chrono::day;

  // Validate invalid values. The range [0, 255] is guaranteed to be allowed.
  assert(testComparisonsValues<day>(0U, 0U));
  assert(testComparisonsValues<day>(0U, 1U));
  assert(testComparisonsValues<day>(254U, 255U));
  assert(testComparisonsValues<day>(255U, 255U));

  // Validate some valid values.
  for (unsigned i = 1; i < 10; ++i)
  {
    for (unsigned j = 1; j < 10; ++j)
    {
      assert(testComparisonsValues<day>(i, j));
    }
  }

  return true;
}

int main(int, char**)
{
  using day = cuda::std::chrono::day;
  AssertComparisonsAreNoexcept<day>();

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  AssertOrderAreNoexcept<day>();
  AssertOrderReturn<cuda::std::strong_ordering, day>();
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  test();
  static_assert(test());

  return 0;
}
