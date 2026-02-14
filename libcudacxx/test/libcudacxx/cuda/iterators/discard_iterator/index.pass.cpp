//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr iter_difference_t<I> offset() const noexcept;

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  {
    const cuda::discard_iterator iter{};
    assert(iter.index() == 0);
    static_assert(cuda::std::is_same_v<decltype(iter.index()), cuda::std::ptrdiff_t>);
  }

  {
    const cuda::std::ptrdiff_t index = 2;
    cuda::discard_iterator iter(index);
    assert(iter.index() == index);
    static_assert(cuda::std::is_same_v<decltype(iter.index()), cuda::std::ptrdiff_t>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
