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

// unique_ptr

// The deleter is not called if get() == 0

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"

class Deleter
{
  int state_;

  __host__ __device__ Deleter(Deleter&);
  __host__ __device__ Deleter& operator=(Deleter&);

public:
  __host__ __device__ TEST_CONSTEXPR_CXX23 Deleter()
      : state_(0)
  {}

  __host__ __device__ TEST_CONSTEXPR_CXX23 int state() const
  {
    return state_;
  }

  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*)
  {
    ++state_;
  }
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  Deleter d;
  assert(d.state() == 0);
  {
    cuda::std::unique_ptr<T, Deleter&> p(nullptr, d);
    assert(p.get() == nullptr);
    assert(&p.get_deleter() == &d);
  }
  assert(d.state() == 0);
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_basic<int>();
  test_basic<int[]>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
