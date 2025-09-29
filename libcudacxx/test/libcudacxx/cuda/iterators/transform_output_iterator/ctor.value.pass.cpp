//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iterator(Fn);
// constexpr explicit iterator(Fn, Integer);

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Fn>
__host__ __device__ constexpr bool test()
{
  int buffer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  Fn func{};

  { // CTAD
    cuda::transform_output_iterator iter{random_access_iterator{buffer + 2}, func};
    assert(base(iter.base()) == buffer + 2);
    *iter = 3;
    assert(buffer[2] == 3 + 1);
    buffer[2] = 2;
#if !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    // The test iterators are not `is_nothrow_move_constructible`
    static_assert(!noexcept(cuda::transform_output_iterator{random_access_iterator{buffer + 2}, func}));
#endif // !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    static_assert(
      cuda::std::is_same_v<decltype(iter), cuda::transform_output_iterator<Fn, random_access_iterator<int*>>>);
  }

  { // CTAD
    cuda::transform_output_iterator iter{buffer + 2, func};
    assert(iter.base() == buffer + 2);
    *iter = 3;
    assert(buffer[2] == 3 + 1);
    buffer[2] = 2;
    static_assert(noexcept(cuda::transform_output_iterator{buffer + 2, func}));
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::transform_output_iterator<Fn, int*>>);
  }

  {
    cuda::transform_output_iterator<Fn, random_access_iterator<int*>> iter{random_access_iterator{buffer + 2}, func};
    assert(base(iter.base()) == buffer + 2);
    *iter = 3;
    assert(buffer[2] == 3 + 1);
    buffer[2] = 2;
#if !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
    // The test iterators are not `is_nothrow_move_constructible`
    static_assert(!noexcept(
      cuda::transform_output_iterator<Fn, random_access_iterator<int*>>{random_access_iterator{buffer + 2}, func}));
#endif // !TEST_COMPILER(GCC, <, 9) && !TEST_COMPILER(MSVC)
  }

  {
    cuda::transform_output_iterator<Fn, int*> iter{buffer + 2, func};
    assert(iter.base() == buffer + 2);
    *iter = 3;
    assert(buffer[2] == 3 + 1);
    buffer[2] = 2;
    static_assert(noexcept(cuda::transform_output_iterator<Fn, int*>{buffer + 2, func}));
  }

  return true;
}

__host__ __device__ constexpr bool test()
{
  test<PlusOne>();
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test<PlusOneHost>();), (test<PlusOneDevice>();))

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
