//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr iter_difference_t<I> count() const noexcept;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

struct InputOrOutputArchetype
{
  using difference_type = int;

  int* ptr;

  __host__ __device__ constexpr int operator*()
  {
    return *ptr;
  }
  __host__ __device__ constexpr void operator++(int)
  {
    ++ptr;
  }
  __host__ __device__ constexpr InputOrOutputArchetype& operator++()
  {
    ++ptr;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
    {
      assert(iter.count() == i);
    }

    static_assert(noexcept(iter.count()));
  }
  {
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
    {
      assert(iter.count() == i);
    }

    static_assert(noexcept(iter.count()));
  }
  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 8; i != 0; --i, ++iter)
    {
      assert(iter.count() == i);
    }
  }
  {
    cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter.count() == 6);
  }

  // Const tests.
  {
    const cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(iter.count() == 8);
  }
  {
    const cuda::std::counted_iterator iter(forward_iterator<int*>{buffer + 1}, 7);
    assert(iter.count() == 7);
  }
  {
    const cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(iter.count() == 6);
  }
  {
    const cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(iter.count() == 6);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
