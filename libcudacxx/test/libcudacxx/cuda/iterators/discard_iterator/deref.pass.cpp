//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr decltype(auto) operator*();
// constexpr decltype(auto) operator*() const
//   requires dereferenceable<const I>;

#include <cuda/iterator>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    const int index = 2;
    cuda::discard_iterator iter(index);
    *iter = 42;
  }

  {
    const int index = 2;
    const cuda::discard_iterator iter(index);
    *iter = 42;
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
