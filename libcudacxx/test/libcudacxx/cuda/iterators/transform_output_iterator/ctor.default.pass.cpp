//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<Fn> = default;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::transform_output_iterator<PlusOne, random_access_iterator<int*>> iter;
    assert(iter.base() == random_access_iterator<int*>{});
  }

  {
    const cuda::transform_output_iterator<PlusOne, random_access_iterator<int*>> iter;
    assert(iter.base() == random_access_iterator<int*>{});
  }

  {
    static_assert(!cuda::std::is_default_constructible_v<
                  cuda::transform_output_iterator<NotDefaultConstructiblePlusOne, random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
