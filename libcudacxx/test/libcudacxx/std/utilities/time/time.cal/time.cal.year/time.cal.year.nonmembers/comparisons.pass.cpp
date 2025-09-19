//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr strong_ordering operator<=>(const year& x, const year& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  using year = cuda::std::chrono::year;

  // Validate valid value. The range [-32768, 32767] is guaranteed to be allowed.
  assert(testComparisonsValues<year>(-32768, -32768));
  assert(testComparisonsValues<year>(-32768, -32767));
  // Largest positive
  assert(testComparisonsValues<year>(32767, 32766));
  assert(testComparisonsValues<year>(32767, 32767));

  // Validate some valid values.
  for (int i = 1; i < 10; ++i)
  {
    for (int j = 1; j < 10; ++j)
    {
      assert(testComparisonsValues<year>(i, j));
    }
  }

  return true;
}

int main(int, char**)
{
  using year = cuda::std::chrono::year;
  AssertComparisonsAreNoexcept<year>();

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  AssertOrderAreNoexcept<year>();
  AssertOrderReturn<cuda::std::strong_ordering, year>();
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  test();
  static_assert(test());

  return 0;
}
