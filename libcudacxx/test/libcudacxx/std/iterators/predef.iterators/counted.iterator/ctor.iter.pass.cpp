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

// constexpr counted_iterator() requires default_initializable<I> = default;
// constexpr counted_iterator(I x, iter_difference_t<I> n);
// template<class I2>
//   requires convertible_to<const I2&, I>
//     constexpr counted_iterator(const counted_iterator<I2>& x);

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
    assert(base(iter.base()) == buffer);
    assert(iter.count() == 8);
  }

  {
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    assert(iter.base() == forward_iterator<int*>{buffer});
    assert(iter.count() == 8);
  }

  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    assert(iter.base() == contiguous_iterator<int*>{buffer});
    assert(iter.count() == 8);
  }

  {
    cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);
    assert(iter.base().ptr == buffer);
    assert(iter.count() == 8);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
