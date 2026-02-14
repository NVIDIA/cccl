//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr discard_iterator operator+(iter_difference_t<I> n) const;
// friend constexpr discard_iterator operator+(iter_difference_t<I> n, const discard_iterator& x);
// constexpr discard_iterator& operator+=(iter_difference_t<I> n);

#include <cuda/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  { // operator+(iter_difference_t<I> n)
    {
      const int index = 2;
      const int diff  = 3;
      cuda::discard_iterator iter(index);
      assert(iter + diff == cuda::discard_iterator(index + diff));
      assert(iter + 0 == cuda::discard_iterator(index));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), cuda::discard_iterator>);
    }

    {
      const int index = 2;
      const int diff  = 3;
      const cuda::discard_iterator iter(index);
      assert(iter + diff == cuda::discard_iterator(index + diff));
      assert(iter + 0 == cuda::discard_iterator(index));

      static_assert(cuda::std::is_same_v<decltype(iter + 2), cuda::discard_iterator>);
    }
  }

  { // operator+(iter_difference_t<I> n, const discard_iterator& x)
    {
      const int index = 2;
      const int diff  = 3;
      cuda::discard_iterator iter(index);
      assert(diff + iter == cuda::discard_iterator(index + diff));
      assert(0 + iter == cuda::discard_iterator(index));

      static_assert(cuda::std::is_same_v<decltype(2 + iter), cuda::discard_iterator>);
    }
  }

  { // operator+=(iter_difference_t<I> n)
    {
      const int index = 2;
      const int diff  = 3;
      cuda::discard_iterator iter(index);
      assert((iter += 0) == cuda::discard_iterator(index));
      assert((iter += diff) == cuda::discard_iterator(index + diff));

      static_assert(cuda::std::is_same_v<decltype(iter += 2), cuda::discard_iterator&>);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
