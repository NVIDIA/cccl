//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// constexpr auto operator->() const noexcept
//   requires contiguous_iterator<I>;

#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
_CCCL_CONCEPT ArrowEnabled = _CCCL_REQUIRES_EXPR((Iter), Iter& iter)(unused(iter.operator->()));

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 0; i < 8; ++i, ++iter)
    {
      assert(iter.operator->() == buffer + i);
    }

    static_assert(noexcept(iter.operator->()));
  }
  {
    const cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter.operator->() == buffer);

    static_assert(noexcept(iter.operator->()));
  }

  {
    static_assert(ArrowEnabled<cuda::std::counted_iterator<contiguous_iterator<int*>>>);
    static_assert(!ArrowEnabled<cuda::std::counted_iterator<random_access_iterator<int*>>>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
