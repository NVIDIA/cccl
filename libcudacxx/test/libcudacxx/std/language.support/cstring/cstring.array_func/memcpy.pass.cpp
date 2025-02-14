//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// void* memcpy(void* dst, const void* src, size_t count);

#include <cuda/std/cassert>
#include <cuda/std/cstring>

#include "test_macros.h"

template <typename T>
__host__ __device__ void test(T obj)
{
  unsigned char buf[sizeof(T)]{};
  assert(cuda::std::memcpy(buf, &obj, sizeof(T)) == buf);
  assert(cuda::std::memcmp(buf, &obj, sizeof(T)) == 0);
}

struct SmallObj
{
  int i;
};

struct MidObj
{
  char c1;
  char c2;
  short s;
  int j;
  int k;
};

struct LargeObj
{
  double ds[10];
};

union Union
{
  int i;
  double d;
};

int main(int, char**)
{
  test('a');
  test(short(2489));
  test(780581);
  test(127156178992l);
  test(129.f);
  test(20123.003445);
  test(SmallObj{25});
  test(MidObj{'a', 'b', 120, 902183, 3124});
  test(LargeObj{187.0, 0.00000346, 1203980985.4365, 123.567, 0.0, -0.0, 123.567});
  test(reinterpret_cast<char*>(123456));
  test(Union{123456});

  return 0;
}
