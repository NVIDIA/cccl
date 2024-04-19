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

// template <class ForwardIt>
// void uninitialized_default_construct(ForwardIt, ForwardIt);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstdlib>

#include "test_iterators.h"
#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR int Counted_count       = 0;
STATIC_TEST_GLOBAL_VAR int Counted_constructed = 0;
struct Counted
{
  __host__ __device__ static void reset()
  {
    Counted_count = Counted_constructed = 0;
  }
  __host__ __device__ explicit Counted()
  {
    ++Counted_count;
    ++Counted_constructed;
  }
  __host__ __device__ Counted(Counted const&)
  {
    assert(false);
  }
  __host__ __device__ ~Counted()
  {
    assert(Counted_count > 0);
    --Counted_count;
  }
  __host__ __device__ friend void operator&(Counted) = delete;
};

STATIC_TEST_GLOBAL_VAR int ThrowsCounted_count       = 0;
STATIC_TEST_GLOBAL_VAR int ThrowsCounted_constructed = 0;
STATIC_TEST_GLOBAL_VAR int ThrowsCounted_throw_after = 0;
struct ThrowsCounted
{
  __host__ __device__ static void reset()
  {
    ThrowsCounted_count = ThrowsCounted_constructed = ThrowsCounted_throw_after = 0;
  }
  __host__ __device__ explicit ThrowsCounted(int&& x)
  {
    ++ThrowsCounted_constructed;
    if (ThrowsCounted_throw_after > 0 && --ThrowsCounted_throw_after == 0)
    {
      TEST_THROW(1);
    }
    ++ThrowsCounted_count;
    x = 0;
  }
  __host__ __device__ ThrowsCounted(ThrowsCounted const&)
  {
    assert(false);
  }
  __host__ __device__ ~ThrowsCounted()
  {
    assert(ThrowsCounted_count > 0);
    --ThrowsCounted_count;
  }
  __host__ __device__ friend void operator&(ThrowsCounted) = delete;
};

__host__ __device__ void test_ctor_throws()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  using It                                                    = forward_iterator<ThrowsCounted*>;
  const int N                                                 = 5;
  alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted) * N] = {};
  ThrowsCounted* p                                            = (ThrowsCounted*) pool;
  try
  {
    ThrowsCounted_throw_after = 4;
    cuda::std::uninitialized_default_construct_n(It(p), N);
    assert(false);
  }
  catch (...)
  {}
  assert(ThrowsCounted_count == 0);
  assert(ThrowsCounted_constructed == 4); // Fourth construction throws
#endif
}

__host__ __device__ void test_counted()
{
  using It                                        = forward_iterator<Counted*>;
  const int N                                     = 5;
  alignas(Counted) char pool[sizeof(Counted) * N] = {};
  Counted* p                                      = (Counted*) pool;
  It e                                            = cuda::std::uninitialized_default_construct_n(It(p), 1);
  assert(e == It(p + 1));
  assert(Counted_count == 1);
  assert(Counted_constructed == 1);
  e = cuda::std::uninitialized_default_construct_n(It(p + 1), 4);
  assert(e == It(p + N));
  assert(Counted_count == 5);
  assert(Counted_constructed == 5);
  cuda::std::__destroy(p, p + N);
  assert(Counted_count == 0);
}

int main(int, char**)
{
  test_counted();
  test_ctor_throws();

  return 0;
}
