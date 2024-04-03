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

// template <class ForwardIterator, class Size, class T>
//   ForwardIterator
//   uninitialized_fill_n(ForwardIterator first, Size n, const T& x);

#include <cuda/std/__memory>
#include <cuda/std/cassert>

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
    assert(B_population == 0);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
      cuda::std::uninitialized_fill_n(bp, 5, B());
      assert(false);
    }
    catch (...)
    {
      assert(B_population == 0);
    }
#endif
    B_count = 0;
    B* r    = cuda::std::uninitialized_fill_n(bp, 2, B());
    assert(r == bp + 2);
    for (int i = 0; i < 2; ++i)
    {
      assert(bp[i].data_ == 1);
    }
    assert(B_population == 2);
  }
  {
    {
      const int N                  = 5;
      char pool[N * sizeof(Nasty)] = {0};
      Nasty* bp                    = (Nasty*) pool;

      Nasty_count = 23;
      cuda::std::uninitialized_fill_n(bp, N, Nasty());
      for (int i = 0; i < N; ++i)
      {
        assert(bp[i].i_ == 23);
      }
    }
  }

  return 0;
}
