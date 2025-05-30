//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// iterator() requires default_initializable<W> = default;

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::counting_iterator<int> iter;
    assert(*iter == 0);
  }

  {
    cuda::counting_iterator<Int42<DefaultTo42>> iter;
    assert((*iter).value_ == 42);
  }

  {
    static_assert(!cuda::std::is_default_constructible_v<cuda::counting_iterator<Int42<ValueCtor>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
