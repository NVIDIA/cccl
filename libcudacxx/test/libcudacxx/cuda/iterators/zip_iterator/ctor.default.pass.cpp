//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// iterator() = default;

#include <cuda/iterator>
#include <cuda/std/tuple>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  {
    cuda::zip_iterator<PODIter> iter;
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }

  {
    cuda::zip_iterator<PODIter> iter{};
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }

  static_assert(!cuda::std::is_default_constructible_v<cuda::zip_iterator<PODIter, IterNotDefaultConstructible>>);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
