//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test rel_ops

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct A
{
  int data_;

  __host__ __device__ explicit A(int data = -1)
      : data_(data)
  {}
};

inline __host__ __device__ bool operator==(const A& x, const A& y)
{
  return x.data_ == y.data_;
}

inline __host__ __device__ bool operator<(const A& x, const A& y)
{
  return x.data_ < y.data_;
}

int main(int, char**)
{
  using namespace cuda::std::rel_ops;
  A a1(1);
  A a2(2);
  assert(a1 == a1);
  assert(a1 != a2);
  assert(a1 < a2);
  assert(a2 > a1);
  assert(a1 <= a1);
  assert(a1 <= a2);
  assert(a2 >= a2);
  assert(a2 >= a1);

  return 0;
}
