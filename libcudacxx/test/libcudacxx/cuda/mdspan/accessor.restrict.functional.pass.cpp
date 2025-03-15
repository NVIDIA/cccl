//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>

#include "test_macros.h"

using dim = cuda::std::dims<1>;

__host__ __device__ void
compute(cuda::restrict_mdspan<int, dim> a, cuda::restrict_mdspan<int, dim> b, cuda::restrict_mdspan<int, dim> c)
{
  c[0] = a[0] * b[0];
  c[1] = a[0] * b[0];
  c[2] = a[0] * b[0] * a[1];
  c[3] = a[0] * a[1];
  c[4] = a[0] * b[0];
  c[5] = b[0];
}

__host__ __device__ void test()
{
  int arrayA[] = {1, 2};
  int arrayB[] = {5};
  int arrayC[] = {9, 10, 11, 12, 13, 14};
  cuda::std::mdspan<int, dim> mdA{arrayA, dim{2}};
  cuda::std::mdspan<int, dim> mdB{arrayB, dim{1}};
  cuda::std::mdspan<int, dim> mdC{arrayC, dim{6}};
  compute(mdA, mdB, mdC);
  assert(mdC[0] == 5);
  assert(mdC[1] == 5);
  assert(mdC[2] == 10);
  assert(mdC[3] == 2);
  assert(mdC[4] == 5);
  assert(mdC[5] == 5);
}

int main(int, char**)
{
  test();
  return 0;
}
