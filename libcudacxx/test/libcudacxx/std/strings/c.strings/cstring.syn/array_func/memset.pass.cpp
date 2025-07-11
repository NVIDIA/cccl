//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// void* memset(void* s, int c, size_t n);

#include <cuda/std/cassert>
#include <cuda/std/cstring>

#include "test_macros.h"

template <typename T>
__host__ __device__ void test(int c)
{
  T obj{};
  assert(cuda::std::memset(&obj, c, sizeof(T)) == &obj);
  assert(cuda::std::memcmp(&obj, &obj, sizeof(T)) == 0);
}

struct SmallObj
{
  int i;
};

struct MidObj
{
  char i;
  int j;
  int k;
};

struct LargeObj
{
  short j;
  double ds[10];
};

int main(int, char**)
{
  test<char>(0);
  test<short>(255);
  test<int>(78);
  test<long>(127);
  test<float>(129);
  test<double>(200);
  test<SmallObj>(25);
  test<MidObj>(100);
  test<LargeObj>(187);
  test<void*>(0);
  test<int[10]>(2);

  return 0;
}
