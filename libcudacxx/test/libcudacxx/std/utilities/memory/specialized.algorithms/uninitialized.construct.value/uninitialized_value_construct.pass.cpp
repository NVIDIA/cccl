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
// void uninitialized_value_construct(ForwardIt, ForwardIt);

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

#ifndef TEST_HAS_NO_EXCEPTIONS
static int ThrowsCounted_count       = 0;
static int ThrowsCounted_constructed = 0;
static int ThrowsCounted_throw_after = 0;
struct ThrowsCounted
{
  static void reset()
  {
    ThrowsCounted_count = ThrowsCounted_constructed = ThrowsCounted_throw_after = 0;
  }
  explicit ThrowsCounted()
  {
    ++ThrowsCounted_constructed;
    if (ThrowsCounted_throw_after > 0 && --ThrowsCounted_throw_after == 0)
    {
      TEST_THROW(1);
    }
    ++ThrowsCounted_count;
  }
  ThrowsCounted(ThrowsCounted const&)
  {
    assert(false);
  }
  ~ThrowsCounted()
  {
    assert(ThrowsCounted_count > 0);
    --ThrowsCounted_count;
  }
  friend void operator&(ThrowsCounted) = delete;
};

void test_ctor_throws()
{
  using It                                                    = forward_iterator<ThrowsCounted*>;
  const int N                                                 = 5;
  alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted) * N] = {};
  ThrowsCounted* p                                            = (ThrowsCounted*) pool;
  try
  {
    ThrowsCounted_throw_after = 4;
    cuda::std::uninitialized_value_construct(It(p), It(p + N));
    assert(false);
  }
  catch (...)
  {}
  assert(ThrowsCounted_count == 0);
  assert(ThrowsCounted_constructed == 4); // forth construction throws
}
#endif // !TEST_HAS_NO_EXCEPTIONS

__host__ __device__ void test_counted()
{
  using It                                        = forward_iterator<Counted*>;
  const int N                                     = 5;
  alignas(Counted) char pool[sizeof(Counted) * N] = {};
  Counted* p                                      = (Counted*) pool;
  cuda::std::uninitialized_value_construct(It(p), It(p + 1));
  assert(Counted_count == 1);
  assert(Counted_constructed == 1);
  cuda::std::uninitialized_value_construct(It(p + 1), It(p + N));
  assert(Counted_count == 5);
  assert(Counted_constructed == 5);
  cuda::std::__destroy(p, p + N);
  assert(Counted_count == 0);
}

__host__ __device__ void test_value_initialized()
{
  using It    = forward_iterator<int*>;
  const int N = 5;
  int pool[N] = {-1, -1, -1, -1, -1};
  int* p      = pool;
  cuda::std::uninitialized_value_construct(It(p), It(p + 1));
  assert(pool[0] == 0);
  assert(pool[1] == -1);
  cuda::std::uninitialized_value_construct(It(p + 1), It(p + N));
  assert(pool[1] == 0);
  assert(pool[2] == 0);
  assert(pool[3] == 0);
  assert(pool[4] == 0);
}

int main(int, char**)
{
  test_counted();
  test_value_initialized();
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_ctor_throws();))
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
