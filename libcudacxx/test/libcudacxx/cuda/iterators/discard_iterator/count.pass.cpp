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
    assert(iter.count() == 0);
    static_assert(cuda::std::is_same_v<decltype(iter.count()), cuda::std::ptrdiff_t>);
  }

  {
    const cuda::std::ptrdiff_t count = 2;
    cuda::discard_iterator iter(count);
    assert(iter.count() == count);
    static_assert(cuda::std::is_same_v<decltype(iter.count()), cuda::std::ptrdiff_t>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
