//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class InputIterator, class ForwardIterator>
//   ForwardIterator
//   uninitialized_copy(InputIterator first, InputIterator last,
//                      ForwardIterator result);

#include <cuda/std/__memory>
#include <cuda/std/cassert>

#include "../overload_compare_iterator.h"
#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR int B_count      = 0;
STATIC_TEST_GLOBAL_VAR int B_population = 0;
struct B
{
  int data_;
  __host__ __device__ explicit B()
      : data_(1)
  {
    ++B_population;
  }
  __host__ __device__ B(const B& b)
  {
    ++B_count;
    if (B_count == 3)
    {
      TEST_THROW(1);
    }
    data_ = b.data_;
    ++B_population;
  }
  __host__ __device__ ~B()
  {
    data_ = 0;
    --B_population;
  }
};

STATIC_TEST_GLOBAL_VAR int Nasty_count = 0;
struct Nasty
{
  __host__ __device__ Nasty()
      : i_(Nasty_count++)
  {}
  __host__ __device__ Nasty* operator&() const
  {
    return nullptr;
  }
  int i_;
};

int main(int, char**)
{
  {
    const int N              = 5;
    char pool[sizeof(B) * N] = {0};
    B* bp                    = (B*) pool;
    B b[N];
    assert(B_population == N);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
      cuda::std::uninitialized_copy(b, b + N, bp);
      assert(false);
    }
    catch (...)
    {
      assert(B_population == N);
    }
#endif
    B_count = 0;
    cuda::std::uninitialized_copy(b, b + 2, bp);
    for (int i = 0; i < 2; ++i)
    {
      assert(bp[i].data_ == 1);
    }
    assert(B_population == N + 2);
  }

  {
    const int N                  = 5;
    char pool[sizeof(Nasty) * N] = {0};
    Nasty* p                     = (Nasty*) pool;
    Nasty arr[N];
    cuda::std::uninitialized_copy(arr, arr + N, p);
    for (int i = 0; i < N; ++i)
    {
      assert(arr[i].i_ == i);
      assert(p[i].i_ == i);
    }
  }

  // Test with an iterator that overloads operator== and operator!= as the input and output iterators
  {
    using T        = int;
    using Iterator = overload_compare_iterator<T*>;
    const int N    = 5;

    // input
    {
      char pool[sizeof(T) * N] = {0};
      T* p                     = reinterpret_cast<T*>(pool);
      T array[N]               = {1, 2, 3, 4, 5};
      cuda::std::uninitialized_copy(Iterator(array), Iterator(array + N), p);
      for (int i = 0; i != N; ++i)
      {
        assert(array[i] == p[i]);
      }
    }

    // output
    {
      char pool[sizeof(T) * N] = {0};
      T* p                     = reinterpret_cast<T*>(pool);
      T array[N]               = {1, 2, 3, 4, 5};
      cuda::std::uninitialized_copy(array, array + N, Iterator(p));
      for (int i = 0; i != N; ++i)
      {
        assert(array[i] == p[i]);
      }
    }
  }

  return 0;
}
