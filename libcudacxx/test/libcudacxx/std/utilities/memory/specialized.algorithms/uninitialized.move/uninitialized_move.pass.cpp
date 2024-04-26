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

// template <class InputIt, class ForwardIt>
// ForwardIt uninitialized_move(InputIt, InputIt, ForwardIt);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstdlib>

#include "../overload_compare_iterator.h"
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
  __host__ __device__ explicit Counted(int&& x)
      : value(x)
  {
    x = 0;
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
  int value;
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
  explicit ThrowsCounted(int&& x)
  {
    ++ThrowsCounted_constructed;
    if (ThrowsCounted_throw_after > 0 && --ThrowsCounted_throw_after == 0)
    {
      TEST_THROW(1);
    }
    ++ThrowsCounted_count;
    x = 0;
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
  int values[N]                                               = {1, 2, 3, 4, 5};
  alignas(ThrowsCounted) char pool[sizeof(ThrowsCounted) * N] = {};
  ThrowsCounted* p                                            = (ThrowsCounted*) pool;
  try
  {
    ThrowsCounted_throw_after = 4;
    cuda::std::uninitialized_move(values, values + N, It(p));
    assert(false);
  }
  catch (...)
  {}
  assert(ThrowsCounted_count == 0);
  assert(ThrowsCounted_constructed == 4); // forth construction throws
  assert(values[0] == 0);
  assert(values[1] == 0);
  assert(values[2] == 0);
  assert(values[3] == 4);
  assert(values[4] == 5);
}
#endif // !TEST_HAS_NO_EXCEPTIONS

__host__ __device__ void test_counted()
{
  using It                                        = cpp17_input_iterator<int*>;
  using FIt                                       = forward_iterator<Counted*>;
  const int N                                     = 5;
  int values[N]                                   = {1, 2, 3, 4, 5};
  alignas(Counted) char pool[sizeof(Counted) * N] = {};
  Counted* p                                      = (Counted*) pool;
  auto ret                                        = cuda::std::uninitialized_move(It(values), It(values + 1), FIt(p));
  assert(ret == FIt(p + 1));
  assert(Counted_constructed == 1);
  assert(Counted_count == 1);
  assert(p[0].value == 1);
  assert(values[0] == 0);
  ret = cuda::std::uninitialized_move(It(values + 1), It(values + N), FIt(p + 1));
  assert(ret == FIt(p + N));
  assert(Counted_count == 5);
  assert(Counted_constructed == 5);
  assert(p[1].value == 2);
  assert(p[2].value == 3);
  assert(p[3].value == 4);
  assert(p[4].value == 5);
  assert(values[1] == 0);
  assert(values[2] == 0);
  assert(values[3] == 0);
  assert(values[4] == 0);
  cuda::std::__destroy(p, p + N);
  assert(Counted_count == 0);
}

int main(int, char**)
{
  test_counted();
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_ctor_throws();))
#endif // !TEST_HAS_NO_EXCEPTIONS

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
      cuda::std::uninitialized_move(Iterator(array), Iterator(array + N), p);
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
      cuda::std::uninitialized_move(array, array + N, Iterator(p));
      for (int i = 0; i != N; ++i)
      {
        assert(array[i] == p[i]);
      }
    }
  }

  return 0;
}
