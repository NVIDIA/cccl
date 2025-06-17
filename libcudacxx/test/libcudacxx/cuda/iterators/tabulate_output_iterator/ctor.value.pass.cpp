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

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  basic_functor func{};

  { // CTAD
    cuda::tabulate_output_iterator iter{func};
    assert(iter.index() == 0);
    *iter = 0;
  }

  { // CTAD
    const int val = 42;
    cuda::tabulate_output_iterator iter{func, val};
    assert(iter.index() == val);
    *iter = val;
  }

  { // CTAD
    cuda::tabulate_output_iterator iter{func, 42};
    assert(iter.index() == 42);
    *iter = 42;
  }

  {
    cuda::tabulate_output_iterator<basic_functor, int> iter{func};
    assert(iter.index() == 0);
    *iter = 0;
  }

  {
    const int val = 42;
    cuda::tabulate_output_iterator<basic_functor, int> iter{func, val};
    assert(iter.index() == val);
    *iter = val;
  }

  {
    cuda::tabulate_output_iterator<basic_functor, int> iter{func, 42};
    assert(iter.index() == 42);
    *iter = 42;
  }

  {
    const short val = 42;
    cuda::tabulate_output_iterator<basic_functor, int> iter{func, val};
    assert(iter.index() == val);
    *iter = val;
  }

  {
    const cuda::std::ptrdiff_t val = 42;
    cuda::tabulate_output_iterator<basic_functor, int> iter{func, val};
    assert(iter.index() == static_cast<int>(val));
    *iter = static_cast<int>(val);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
