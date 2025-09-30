//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month;

// constexpr bool operator==(const month& x, const month& y) noexcept;
// constexpr strong_ordering operator<=>(const month& x, const month& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using month = cuda::std::chrono::month;

  // Validate invalid values. The range [0, 255] is guaranteed to be allowed.
  assert(testComparisonsValues<month>(0U, 0U));
  assert(testComparisonsValues<month>(0U, 1U));
  assert(testComparisonsValues<month>(254U, 255U));
  assert(testComparisonsValues<month>(255U, 255U));

  // Validate some valid values.
  for (unsigned i = 1; i <= 12; ++i)
  {
    for (unsigned j = 1; j <= 12; ++j)
    {
      assert(testComparisonsValues<month>(i, j));
    }
  }

  return true;
}

int main(int, char**)
{
  using month = cuda::std::chrono::month;
  AssertComparisonsAreNoexcept<month>();

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  AssertOrderAreNoexcept<month>();
  AssertOrderReturn<cuda::std::strong_ordering, month>();
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  test();
  static_assert(test());

  return 0;
}
