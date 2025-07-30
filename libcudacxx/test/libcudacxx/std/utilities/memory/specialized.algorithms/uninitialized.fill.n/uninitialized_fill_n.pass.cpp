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

#include <cuda/std/cassert>
#include <cuda/std/memory>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int Nasty_count = 0;
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

#if TEST_HAS_EXCEPTIONS()
static int B_count      = 0;
static int B_population = 0;
struct B
{
  int data_;
  explicit B()
      : data_(1)
  {
    ++B_population;
  }
  B(const B& b)
  {
    ++B_count;
    if (B_count == 3)
    {
      TEST_THROW(1);
    }
    data_ = b.data_;
    ++B_population;
  }
  ~B()
  {
    data_ = 0;
    --B_population;
  }
};

void test_exceptions()
{
  const int N              = 5;
  char pool[sizeof(B) * N] = {0};
  B* bp                    = (B*) pool;
  assert(B_population == 0);
  try
  {
    cuda::std::uninitialized_fill_n(bp, 5, B());
    assert(false);
  }
  catch (...)
  {
    assert(B_population == 0);
  }
  B_count = 0;
  B* r    = cuda::std::uninitialized_fill_n(bp, 2, B());
  assert(r == bp + 2);
  for (int i = 0; i < 2; ++i)
  {
    assert(bp[i].data_ == 1);
  }
  assert(B_population == 2);
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
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
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
