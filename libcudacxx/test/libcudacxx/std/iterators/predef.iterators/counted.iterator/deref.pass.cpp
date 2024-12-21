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

// constexpr decltype(auto) operator*();
// constexpr decltype(auto) operator*() const
//   requires dereferenceable<const I>;

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

struct InputOrOutputArchetype
{
  using difference_type = int;

  int* ptr;

  __host__ __device__ constexpr int operator*() const
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

struct NonConstDeref
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
  __host__ __device__ constexpr NonConstDeref& operator++()
  {
    ++ptr;
    return *this;
  }
};

#if TEST_STD_VER >= 2020
template <class T>
concept IsDereferenceable = requires(T& i) { *i; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class, class = void>
inline constexpr bool IsDereferenceable = false;

template <class Iter>
inline constexpr bool IsDereferenceable<Iter, cuda::std::void_t<decltype(*cuda::std::declval<Iter&>())>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    static_assert(IsDereferenceable<cuda::std::counted_iterator<InputOrOutputArchetype>>);
    static_assert(IsDereferenceable<const cuda::std::counted_iterator<InputOrOutputArchetype>>);
    static_assert(IsDereferenceable<cuda::std::counted_iterator<NonConstDeref>>);
    static_assert(!IsDereferenceable<const cuda::std::counted_iterator<NonConstDeref>>);
  }

  {
    cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
    {
      assert(*iter == i);
    }
  }

  {
    cuda::std::counted_iterator iter(forward_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
    {
      assert(*iter == i);
    }
  }

  {
    cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
    {
      assert(*iter == i);
    }
  }

  {
    cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer}, 8);
    for (int i = 1; i < 9; ++i, ++iter)
    {
      assert(*iter == i);
    }
  }

  {
    const cuda::std::counted_iterator iter(cpp20_input_iterator<int*>{buffer}, 8);
    assert(*iter == 1);
  }

  {
    const cuda::std::counted_iterator iter(forward_iterator<int*>{buffer + 1}, 7);
    assert(*iter == 2);
  }

  {
    const cuda::std::counted_iterator iter(contiguous_iterator<int*>{buffer + 2}, 6);
    assert(*iter == 3);
  }

  {
    const cuda::std::counted_iterator iter(InputOrOutputArchetype{buffer + 2}, 6);
    assert(*iter == 3);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
