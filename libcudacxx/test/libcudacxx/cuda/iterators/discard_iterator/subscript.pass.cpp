//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator[](iter_difference_t<I> n) const;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::discard_iterator iter{};
    iter[42] = 1337;
  }

  {
    const cuda::discard_iterator iter{};
    iter[42] = 1337;
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
