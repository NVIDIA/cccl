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
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    auto iter = cuda::make_discard_iterator();
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::discard_iterator>);
    *iter = 42;
  }

  {
    auto iter = cuda::make_discard_iterator(static_cast<short>(42));
    static_assert(cuda::std::is_same_v<decltype(iter), cuda::discard_iterator>);
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
