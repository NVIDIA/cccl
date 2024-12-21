//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr common_iterator() requires default_initializable<I> = default;

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_iterators.h"

__host__ __device__ constexpr bool test()
{
  {
    using It       = cpp17_input_iterator<int*>;
    using CommonIt = cuda::std::common_iterator<It, sentinel_wrapper<It>>;
    static_assert(!cuda::std::is_default_constructible_v<It>); // premise
    static_assert(!cuda::std::is_default_constructible_v<CommonIt>); // conclusion
  }
  {
    // The base iterator is value-initialized.
    using CommonIt = cuda::std::common_iterator<int*, sentinel_wrapper<int*>>;
    CommonIt c;
    assert(c == static_cast<CommonIt>(nullptr));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
